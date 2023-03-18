from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __int__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train() # declare train mode

        # Shuffle before begin
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradient of the model.

            self.optimizer.zero_grad() # grad 초기화
            # optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정합니다.
            # 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.

            loss_i.backward()
            # loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파합니다.
            # PyTorch는 각 매개변수에 대한 손실의 변화도를 저장합니다.

            self.optimizer.step()
            # 변화도를 계산한 뒤에는 optimizer.step()을 호출하여
            # 역전파 단계에서 수집된 변화도로 매개변수를 조정합니다.

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on
        self.model.eval()

        # Turn on the no_grad mode to make more efficiently.

        with torch.no_grad():
        # 자원을 획득하고 사용 후 반납해야 하는 경우 주로 사용합니다.
        # 1) 자원을 획득한다 2) 자원을 사용한다 3) 자원을 반납한다

            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            # Returns a random permutation of integers from 0 to n - 1.
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf # + 무한대
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                # state_dict 는 PyTorch에서 모델을 저장하거나 불러오는 데 관심이 있다면 필수적인 항목입니다.
                # Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s state_dict.
                # Optimizer objects (torch.optim) also have a state_dict, which contains information about the optimizer’s state, as well as the hyperparameters used.

            print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss
            ))

            # Restore to best model.
            self.model.load_state_dict(best_model)


