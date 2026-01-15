from .dataset import *
from .models import *
from .utils import *

class Trainer:
    def __init__(self, config_file='config.yaml', model_name='SpliceAI', train_file=None, val_file=None, test_file=None,
                 metrics_file=None, lr_lambda=None, print_every_n_batches=100):
        self.config_dir = f'{resources.files("spredle").parent}/config'
        self.config = self.load_yaml(config_file)

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}', flush=True)
        model_class = eval(self.config['models'][model_name]['class'])
        cfg = Config(self.config['models'][model_name]['params'])
        cfg.device = self.device
        self.model = model_class(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.task == 'classification' :
            self.loss_fn = CustomLoss(n_classes=cfg.n_classes)
        elif cfg.task == 'regression':
            self.loss_fn = CustomLossReg(n_regs=cfg.n_regs)
        elif cfg.task == 'classification+regression':
            self.loss_fn = CustomLossClsReg(n_classes=cfg.n_classes, n_regs=cfg.n_regs)

        self.print_every_n_batches = print_every_n_batches

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if train_file and os.path.exists(train_file):
            self.train_dataset = torch.load(train_file, weights_only=False)
        if val_file and os.path.exists(val_file):
            self.val_dataset = torch.load(val_file, weights_only=False)
        if test_file and os.path.exists(test_file):
            self.test_dataset = torch.load(test_file, weights_only=False)

        if not metrics_file:
            self.metrics_file = f'{model_name}_metrics.txt'

        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_confusion = []
        self.val_confusion = []
        self.test_confusion = []

        self.learning_rates = []
        self.lr_scheduler = None
        if lr_lambda:
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_lambda[epoch])

        self.early_stopping = None
        if cfg.early_stopping:
            self.early_stopping = EarlyStopping()
    
        self.best_epochs = []
        self.cfg = cfg

    def train(self, epoch):
        self.model.train()

        loss_total = 0
        for nb, (X, y) in enumerate(self.train_dataset):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss.detach().item()

            if nb % self.print_every_n_batches == 0:
                print(f'train epoch:{epoch} batch:{nb} loss:{loss.detach().item()}')

        self.save_checkpoint(epoch)
        train_loss = loss_total / len(self.train_dataset)
        self.train_loss.append(train_loss)
        self.epochs.append(epoch)

        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.learning_rates.append(lr)
        print(f'train epoch:{epoch} avg loss: {" ".join([str(x) for x in self.train_loss])}')

    def validate(self, epoch, test=False):
        ds = 'val'
        dataset = self.val_dataset
        loss_list = self.val_loss
        if test:
            ds = 'test'
            self.epochs.append(epoch)
            dataset = self.test_dataset
            loss_list = self.test_loss

        self.model.eval()
        loss_total = 0
        with torch.no_grad():
            for nb, (X, y) in enumerate(dataset):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss_total += loss.detach().item()

                if nb % self.print_every_n_batches == 0:
                    print(f'{ds} epoch:{epoch} batch:{nb} loss:{loss.detach().item()}', flush=True)
        loss_avg = loss_total / len(dataset)
        loss_list.append(loss_avg)
        print(f'{ds} epoch:{epoch} avg loss: {" ".join([str(x) for x in loss_list])}')

    def test(self, epoch):
        self.load_checkpoint(epoch)
        self.validate(epoch, test=test)
        self.get_confusion(dataset='test')
        self.log_metrics()

    def get_confusion(self, dataset='val', labels=[0, 1], down_sampling=100):
        self.model.eval()
        if dataset == 'val':
            ds = self.val_dataset
            cml = self.val_confusion
        elif dataset == 'train':
            ds = self.train_dataset
            cml = self.train_confusion
        elif dataset == 'test':
            ds = self.test_dataset
            cml = self.test_confusion

        y_true = []
        y_pred = []
        with torch.no_grad():
            for nb, (X, y) in enumerate(ds):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)

                if self.cfg.n_classes == 2:
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(pred.argmax(dim=1).cpu().numpy())

                elif self.cfg.n_classes == 3:
                    y_true.append(y.cpu().numpy())
                    y_pred.append(pred.cpu().numpy())
                    if down_sampling and nb > down_sampling:
                        break

        if self.cfg.n_classes == 2:
            cm = confusion_matrix(y_true, y_pred, labels=labels).ravel()
            cml.append(','.join([str(x) for x in cm]))
            print(f'{dataset} confusion matrix (tn, fp, fn, tp): {cm}')

        elif self.cfg.n_classes == 3:
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
    
            y_pred = y_pred.transpose(0, 2, 1).reshape(-1, 3)
            y_true = y_true.transpose(0, 2, 1).reshape(-1, 3)
    
            wh = y_true.sum(axis=1) >= 1
            y_true = y_true[wh, :]
            y_pred = y_pred[wh, :]
    
            yt1 = y_true[:, 1]
            yp1 = (y_pred.argmax(axis=1) == 1).astype(int)
    
            yt2 = y_true[:, 2]
            yp2 = (y_pred.argmax(axis=1) == 2).astype(int)
    
            yt12 = (y_true[:, 0] == 0).astype(int)
            yp12 = (y_pred.argmax(axis=1) != 0).astype(int)
    
            cms = []
            for yt,yp,name in [[yt2, yp2, 'donor'], [yt1, yp1, 'acceptor'], [yt12, yp12, 'splice_site']]:
                cm = confusion_matrix(yt, yp, labels=labels).ravel()
                cms.append(','.join([str(x) for x in cm] + [name]))
                print(f'{dataset} {name} confusion matrix (tn, fp, fn, tp): {cm}')
            cml.append(';'.join(cms))

    def predict(self, epoch=None, pred_file='predict.txt', out_file='predicted.txt', alphabet=['N', 'A', 'C', 'G', 'T']):
        self.load_checkpoint(epoch)
        self.model.eval()
        with torch.no_grad():
            with open(pred_file) as inFile, open(out_file, 'w') as ouFile:
                for line in inFile:
                    seq = line.strip()
                    seq2 = torch.tensor([[alphabet.index(x) if x in alphabet else 0 for x in seq]]).to(self.device)
                    pred = self.model(seq2)
                    pred = torch.softmax(pred, dim=1).cpu()
                    clss = torch.argmax(pred, dim=1).squeeze().numpy().astype(str)
                    pad = '.' * self.cfg.flank_size
                    clss = pad + ''.join(clss) + pad
                    ouFile.write(f'{seq}\n{clss}\n')

    def log_metrics(self):
        df = pd.DataFrame()
        cols = ['epoch', 'best_epoch', 'learning_rate',
                      'train_loss', 'val_loss', 'test_loss',
                      'train_confusion', 'val_confusion', 'test_confusion']
        df = pd.DataFrame('.', index=range(len(self.epochs)), columns=cols)

        df['epoch'] = self.epochs
        if self.train_loss:
            df['train_loss'] = self.train_loss
        if self.val_loss:
            df['val_loss'] = self.val_loss
        if self.test_loss:
            df['test_loss'] = self.test_loss

        if self.train_confusion:
            df['train_confusion'] = self.train_confusion
        if self.val_confusion:
            df['val_confusion'] = self.val_confusion
        if self.test_confusion:
            df['test_confusion'] = self.test_confusion

        if self.best_epochs:
            df['best_epoch'] = self.best_epochs
        if self.learning_rates:
            df['learning_rate'] = self.learning_rates

        if os.path.exists(self.metrics_file):
            df.tail(1).to_csv(self.metrics_file, index=False, header=False, sep='\t', mode='a')
        else:
            df.tail(1).to_csv(self.metrics_file, index=False, header=True, sep='\t')

        if self.train_loss and self.val_loss:
            plot_file = self.metrics_file.replace('.txt', '.pdf')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(df['epoch'], df['train_loss'], label='train_loss')
            ax.plot(df['epoch'], df['val_loss'], label='val_loss')
            plt.savefig(plot_file)
            plt.close()

    def count_parameters(self, with_lazy=True, show_details=False):
        if with_lazy:
            X,y = next(iter(self.train_dataset))
            pred = self.model.forward(X)

        L = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                pnum = parameter.numel()
                L.append([name, pnum])
        df = pd.DataFrame(L)
        df.columns = ['name', 'pnum']
        if show_details:
            print(df)
        print(f'Total Trainable Parameters: {df["pnum"].sum()}')

    def set_learning_rate(self, lr=1e-3):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, epoch):
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }
        out_file = f'{self.model_name}_ckpt_{epoch}.pt'
        torch.save(checkpoint, out_file) 

    def load_checkpoint(self, epoch):
        in_file = f'{self.model_name}_ckpt_{epoch}.pt'
        if os.path.exists(in_file):
            print(f'loading checkpoint from {in_file}')
            checkpoint = torch.load(in_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f'checkpoint file {in_file} not found!')

    def load_yaml(self, config_file):
        if not os.path.exists(config_file):
            config_file = self.config_dir + '/' + config_file
            print(f'using config {config_file}')
        config = {}
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f'Error loading config file {e}')

    def run(self, resume_epoch=None, start_epoch=0, end_epoch=10, validate=True):
        if resume_epoch is not None:
            self.load_checkpoint(resume_epoch)
            start_epoch = resume_epoch + 1

        for epoch in range(start_epoch, end_epoch):
            self.train(epoch)
            self.get_confusion(dataset='train')
            if validate:
                self.validate(epoch)
                self.get_confusion(dataset='val')

                if self.early_stopping:
                    self.early_stopping(self.val_loss[-1], epoch)
                    self.best_epochs.append(self.early_stopping.best_epoch)

                if self.early_stopping and self.early_stopping.stopped:
                    print(f'Early stopped, beast epoch: {self.early_stopping.best_epoch}')
                    break

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.log_metrics()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.count_parameters()
    trainer.run()
    trainer.test()
    trainer.predict()
    trainer.saliency_map()
