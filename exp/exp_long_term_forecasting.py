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
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

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
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ============================================================
                # 1. Forward Pass (前向传播)
                # ============================================================
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # ============================================================
                # 2. Output Unpacking (解包 M-Stream 输出)
                # ============================================================
                loss_proxy = torch.tensor(0.0, device=self.device)
                loss_orth = torch.tensor(0.0, device=self.device)
                
                # 检查是否为 M-Stream 返回的元组 (pred, loss_proxy, info)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                    # 如果返回元组长度 > 1，且第二个元素是 Tensor，则认为是 Proxy Loss
                    if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                        loss_proxy = outputs[1]
                else:
                    pred = outputs

                # 裁剪预测结果以匹配 Pred_Len
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # ============================================================
                # 3. Loss Calculation (计算总损失)
                # ============================================================
                # A. 预测 Loss
                loss_pred = criterion(pred, batch_y)

                # B. 正交 Loss (仅当模型有 memory 模块且系数 > 0 时计算)
                lambda_orth = getattr(self.args, 'lambda_orth', 0.0)
                if lambda_orth > 0 and hasattr(self.model, 'memory') and hasattr(self.model.memory, 'compute_orthogonal_loss'):
                    loss_orth = self.model.memory.compute_orthogonal_loss()
                
                # C. 加权求和
                lambda_proxy = getattr(self.args, 'lambda_proxy', 0.0)
                loss = loss_pred + lambda_proxy * loss_proxy + lambda_orth * loss_orth
                
                # 记录日志
                train_loss.append(loss.item())
                train_loss_pred.append(loss_pred.item())
                train_loss_proxy.append(loss_proxy.item())
                train_loss_orth.append(loss_orth.item())

                # ============================================================
                # 4. Logging & Scheduler Step
                # ============================================================
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total: {2:.7f} | pred: {3:.7f} | proxy: {4:.7f} | orth: {5:.7f}".format(
                        i + 1, epoch + 1, loss.item(), loss_pred.item(), loss_proxy.item(), loss_orth.item()))
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # [关键] Scheduler Step 必须在 Batch 循环内 (针对 OneCycleLR)
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                # ============================================================
                # 5. Backward Pass (反向传播)
                # ============================================================
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, RSE: {rse:.6f}, R2: {r2:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}, DTW: {dtw}\n')
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
