import torch

class WarmupStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self,  warmup_steps, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, step_size, gamma=self.gamma, last_epoch=last_epoch)
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        # e.g. warmup_steps = 10, case: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21...
        if self.last_epoch == self.warmup_steps or(self.last_epoch % self.step_size != 0 and self.last_epoch > self.warmup_steps):
            return [group['lr'] for group in self.optimizer.param_groups]
        # e.g. warmup_steps = 10, case: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        elif self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * float(self.last_epoch + 1) / float(self.warmup_steps) for group in self.optimizer.param_groups]
        
        
        # e.g. warmup_steps = 10, case: 10, 20, 30, 40...
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * float(self.last_epoch) / (self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** ((self.last_epoch -  self.warmup_steps)// self.step_size) for base_lr in self.base_lrs]
        
        
if __name__ == '__main__':
    optimizer = torch.optim.Adam([torch.ones(3,3)], lr=0.16)
    scheduler = WarmupStepLR(10, optimizer, 10)
    for i in range(50):
        print(i, ": ", scheduler.get_last_lr()[0])
        scheduler.step()