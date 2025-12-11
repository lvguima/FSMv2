from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        is_titan_stream = (self.args.model == 'TitanStream')
        if is_titan_stream and hasattr(self.model, "reset_memory"):
            self.model.reset_memory()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if is_titan_stream:
                            model_output = self.model(batch_x, use_state=False)
                        else:
                            model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if is_titan_stream:
                        model_output = self.model(batch_x, use_state=False)
                    else:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # v6.0 FIX: Handle MStream returning (pred, enc_out) in offline mode
                if isinstance(model_output, tuple):
                    outputs = model_output[0]
                else:
                    outputs = model_output
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                min_len = min(outputs.shape[1], batch_y.shape[1])
                if min_len <= 0:
                    continue
                if outputs.shape[1] != min_len:
                    outputs = outputs[:, -min_len:, :]
                if batch_y.shape[1] != min_len:
                    batch_y = batch_y[:, -min_len:, :]

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        if is_titan_stream and hasattr(self.model, "reset_memory"):
            self.model.reset_memory()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        is_titan_stream = (self.args.model == 'TitanStream')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # [新增] OneCycleLR 调度器初始化 (TST 模式)
        if self.args.lradj == 'TST':
            print("Using OneCycleLR Scheduler")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=len(train_loader),
                pct_start=getattr(self.args, 'pct_start', 0.3),
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        else:
            scheduler = None

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            # [新增] 辅助 Loss 记录列表
            train_loss_pred = []
            train_loss_proxy = []
            train_loss_orth = []

            self.model.train()
            # [修复] 移除 epoch 级别的记忆复位，改为 batch 级别
            # if is_titan_stream and hasattr(self.model, "reset_memory"):
            #     self.model.reset_memory()
            chunk_len_time = getattr(self.args, "chunk_len", 0)
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # [修复 1.1] 每个 batch 开始时重置记忆（模拟每个序列独立的在线流）
                if is_titan_stream and hasattr(self.model, "reset_memory"):
                    self.model.reset_memory()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp_full = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp_full = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp_full], dim=1).float().to(self.device)

                # 支持按 batch 维度的分块累积，降低显存
                chunk_size = getattr(self.args, "chunk_size", 0)
                use_high_order = getattr(self.args, "use_high_order", False)
                num_chunks = 0
                loss_chunks = []
                loss_pred_vals = []
                loss_proxy_vals = []
                loss_orth_vals = []
                backward_in_slices = False  # True 表示时间片内部已执行 backward（截断高阶）

                batch_indices = [(0, batch_x.size(0))] if chunk_size <= 0 else [
                    (s, min(s + chunk_size, batch_x.size(0))) for s in range(0, batch_x.size(0), chunk_size)
                ]

                for s, e in batch_indices:
                    num_chunks += 1
                    sub_x = batch_x[s:e]
                    sub_y = batch_y[s:e]
                    sub_x_mark = batch_x_mark[s:e]
                    sub_y_mark = batch_y_mark[s:e]
                    dec_inp = dec_inp_full[s:e]

                    # 可选的时间维 chunk（Titan 专用，要求 chunk_len >= pred_len）
                    if is_titan_stream and chunk_len_time > 0 and chunk_len_time >= self.args.pred_len and chunk_len_time < self.args.seq_len:
                        time_slices = []
                        seq_len = sub_x.size(1)
                        for t in range(0, seq_len, chunk_len_time):
                            t_end = min(t + chunk_len_time, seq_len)
                            if t_end - t < self.args.pred_len:
                                break
                            time_slices.append((t, t_end))
                    else:
                        time_slices = [(0, sub_x.size(1))]

                    # ============================================================
                    # 1. Forward Pass
                    # ============================================================
                    chunk_losses = []
                    chunk_loss_preds = []
                    chunk_loss_proxies = []
                    chunk_loss_orths = []

                    # [修复 1.3] 训练时记忆延续：初始化临时记忆状态
                    temp_memory = None
                    temp_momentum = None

                    for t, (t_start, t_end) in enumerate(time_slices):
                        x_slice = sub_x[:, t_start:t_end]
                        y_slice = sub_y[:, t_start:t_end]
                        y_mark_slice = sub_y_mark[:, t_start:t_end] if sub_y_mark is not None else None
                        x_mark_slice = sub_x_mark[:, t_start:t_end] if sub_x_mark is not None else None
                        # decoder input与时间片对齐（防止越界）
                        dec_slice = dec_inp[:, :y_slice.size(1)] if dec_inp is not None else None

                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if is_titan_stream:
                                    # [修复 1.3 + inplace] 第一个 chunk 使用初始记忆，后续 chunk 延续状态
                                    # 通过参数传递而非 inplace 操作
                                    update_state_for_chunk = use_high_order  # 仅在高阶模式下更新
                                    differentiable_update_flag = use_high_order  # 高阶模式使用可微分更新

                                    outputs = self.model(
                                        x_slice,
                                        use_state=False,  # 不使用 buffer
                                        update_state=update_state_for_chunk,
                                        differentiable_update=differentiable_update_flag,
                                        external_memory=temp_memory,  # 传递外部记忆
                                        external_momentum=temp_momentum  # 传递外部动量
                                    )
                                else:
                                    outputs = self.model(x_slice, x_mark_slice, dec_slice, y_mark_slice)
                        else:
                            if is_titan_stream:
                                # [修复 1.3 + inplace] 第一个 chunk 使用初始记忆，后续 chunk 延续状态
                                # 通过参数传递而非 inplace 操作
                                update_state_for_chunk = use_high_order  # 仅在高阶模式下更新
                                differentiable_update_flag = use_high_order  # 高阶模式使用可微分更新

                                outputs = self.model(
                                    x_slice,
                                    use_state=False,  # 不使用 buffer
                                    update_state=update_state_for_chunk,
                                    differentiable_update=differentiable_update_flag,
                                    external_memory=temp_memory,  # 传递外部记忆
                                    external_momentum=temp_momentum  # 传递外部动量
                                )
                            else:
                                outputs = self.model(x_slice, x_mark_slice, dec_slice, y_mark_slice)

                    # ============================================================
                    # 2. Output Unpacking
                    # ============================================================
                        loss_proxy = torch.tensor(0.0, device=self.device)
                        loss_orth = torch.tensor(0.0, device=self.device)

                        if isinstance(outputs, tuple):
                            pred = outputs[0]
                            if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                                loss_proxy = outputs[1]
                            # [修复 1.3] 提取更新后的记忆状态（如果有）
                            if len(outputs) > 2 and isinstance(outputs[2], dict):
                                stats = outputs[2]
                                if stats.get('updated_memory') is not None:
                                    temp_memory = stats['updated_memory']
                                if stats.get('updated_momentum') is not None:
                                    temp_momentum = stats['updated_momentum']
                        else:
                            pred = outputs

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred = pred[:, -self.args.pred_len:, f_dim:]
                        target_slice = y_slice[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 对齐长度，避免异常窗口导致维度不匹配
                        min_len = min(pred.shape[1], target_slice.shape[1])
                        if min_len <= 0:
                            continue
                        if pred.shape[1] != min_len:
                            pred = pred[:, -min_len:, :]
                        if target_slice.shape[1] != min_len:
                            target_slice = target_slice[:, -min_len:, :]

                        # ============================================================
                        # 3. Loss Calculation
                        # ============================================================
                        loss_pred = criterion(pred, target_slice)

                        lambda_orth = getattr(self.args, 'lambda_orth', 0.0)
                        if lambda_orth > 0 and hasattr(self.model, 'memory') and hasattr(self.model.memory, 'compute_orthogonal_loss'):
                            loss_orth = self.model.memory.compute_orthogonal_loss()
                        
                        lambda_proxy = getattr(self.args, 'lambda_proxy', 0.0)
                        loss_total = loss_pred + lambda_proxy * loss_proxy + lambda_orth * loss_orth

                        chunk_loss_preds.append(loss_pred.item())
                        chunk_loss_proxies.append(loss_proxy.item())
                        chunk_loss_orths.append(loss_orth.item())

                        # 截断一阶：时间片内直接反向传播，避免跨片高阶图
                        if (not use_high_order) and len(time_slices) > 1:
                            backward_in_slices = True
                            if self.args.use_amp:
                                scaler.scale(loss_total).backward()
                            else:
                                loss_total.backward()
                        else:
                            chunk_losses.append(loss_total)

                    # 汇总时间片损失
                    if not backward_in_slices:
                        loss_chunk_mean = torch.stack(chunk_losses).mean() if len(chunk_losses) > 1 else chunk_losses[0]
                        loss_chunks.append(loss_chunk_mean)
                    loss_pred_vals.append(sum(chunk_loss_preds) / len(chunk_loss_preds))
                    loss_proxy_vals.append(sum(chunk_loss_proxies) / len(chunk_loss_proxies))
                    loss_orth_vals.append(sum(chunk_loss_orths) / len(chunk_loss_orths))

                loss = torch.stack(loss_chunks).mean() if len(loss_chunks) > 1 else (loss_chunks[0] if loss_chunks else torch.tensor(0.0, device=self.device))

                # 记录日志
                train_loss.append(loss.item())
                train_loss_pred.append(sum(loss_pred_vals) / len(loss_pred_vals))
                train_loss_proxy.append(sum(loss_proxy_vals) / len(loss_proxy_vals))
                train_loss_orth.append(sum(loss_orth_vals) / len(loss_orth_vals))

                # ============================================================
                # 4. Logging & Scheduler Step
                # ============================================================
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total: {2:.7f} | pred: {3:.7f} | proxy: {4:.7f} | orth: {5:.7f}".format(
                        i + 1, epoch + 1, loss.item(), train_loss_pred[-1], train_loss_proxy[-1], train_loss_orth[-1]))
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                # ============================================================
                # 5. Backward Pass
                # ============================================================
                if not backward_in_slices:
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                clip_norm = getattr(self.args, "clip_grad", 0.0)
                if clip_norm > 0:
                    if self.args.use_amp:
                        scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    model_optim.step()

            # End of Epoch Loop
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Step decay scheduler (非 TST 模式)
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join(self.args.test_result_dir, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # v6.0 FIX: Handle MStream returning (pred, enc_out) in offline mode
                if isinstance(model_output, tuple):
                    outputs = model_output[0]  # Only take predictions for test
                else:
                    outputs = model_output

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join(self.args.result_dir, setting)
        os.makedirs(folder_path, exist_ok=True)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # 计算 RSE 和 R2
        from utils.metrics import RSE
        rse = RSE(preds, trues)
        # R2 = 1 - (SSE / SST)
        sse = np.sum((trues - preds) ** 2)
        sst = np.sum((trues - trues.mean()) ** 2)
        r2 = 1 - (sse / sst) if sst > 0 else 0.0
        
        print('\n' + '='*60)
        print('Test Results')
        print('='*60)
        print(f'MSE:  {mse:.6f}')
        print(f'MAE:  {mae:.6f}')
        print(f'RMSE: {rmse:.6f}')
        print(f'RSE:  {rse:.6f}')
        print(f'R2:   {r2:.6f}')
        print(f'MAPE: {mape:.6f}')
        print(f'MSPE: {mspe:.6f}')
        print(f'DTW:  {dtw}')
        print('='*60 + '\n')

        # 使用 with 语句安全地写入结果文件
        result_file = os.path.join(self.args.result_dir, 'result_long_term_forecast.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(f"{setting}\n")
            f.write(f'MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, RSE: {rse:.6f}, R2: {r2:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}, DTW: {dtw}\n\n')

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, rse, r2]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return
