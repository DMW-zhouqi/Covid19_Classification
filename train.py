import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from MakeDataset import makedataset
from torchsummary import summary
from AMResNet import AMResNet

# Parameter Settings
batchsz = 16
lr = 1e-3
epochs = 500

torch.manual_seed(1234)
train_db = makedataset('mydata', 512, mode='train')
val_db = makedataset('mydata', 512, mode='val')
test_db = makedataset('mydata', 512, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)
print('num_train:', len(train_loader.dataset))
print('num_val:', len(val_loader.dataset))
print('num_test:', len(test_loader.dataset))

# define validation set evaluate function
def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

# define test set evaluate function
def test(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    num_Covid19, num_Normal, num_Pneumonia = 0, 0, 0
    num_Covid19_T, num_Normal_T, num_Pneumonia_T = 0, 0, 0

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        # print(y.shape)
        for label in y:
            if label == 0:
                num_Covid19 += 1
            if label == 1:
                num_Normal += 1
            if label == 2:
                num_Pneumonia += 1

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            for i in range(len(pred)):
                for j in range(i, len(y)):
                    if pred[i] == y[j] and y[j] == 0:
                        num_Covid19_T += 1
                    if pred[i] == y[j] and y[j] == 1:
                        num_Normal_T += 1
                    if pred[i] == y[j] and y[j] == 2:
                        num_Pneumonia_T += 1
                    break
        correct += torch.eq(pred, y).sum().float().item()

    print('num_Covid19:', num_Covid19, 'num_Covid19_T:', num_Covid19_T)
    print('num_Normal:', num_Normal, 'num_Normal_T:', num_Normal_T)
    print('num_Pneumonia:', num_Pneumonia, 'num_Pneumonia_T:', num_Pneumonia_T)
    return correct / total

# define train function
def main():
    model = AMResNet()
    print(summary(model, (3, 512, 512)))
    model = model.cuda()
    # x = torch.randn(1, 3, 512, 512)
    # print(model(x).shape)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)

        if epoch % 1 == 0:
            val_acc = test(model, test_loader)
            print('Epoch:', epoch, '/', epochs-1, 'loss:', loss, 'acc_val:', val_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                print(best_acc)

                torch.save(model.state_dict(), 'mydata/weights_AMResNet.mdl')

    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('mydata/weights_AMResNet.mdl'))
    print('loaded from ckpt!')

    # test model performance
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)

'''
model = AMResNet()
model.cuda()
summary(model, (3, 512, 512))
model.load_state_dict(torch.load('mydata/weights_AMResNet.mdl'))
print('loaded from ckpt!')

test_acc = test(model, test_loader)
print('test acc:', test_acc)
'''

if __name__ == '__main__':
    main()
