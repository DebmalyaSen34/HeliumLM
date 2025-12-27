import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor):
        #1. Standard Learning (Hard Targets)
        loss_ce = self.ce_loss(student_logits, targets)

        #2. Distillation Learning (Soft Targets)
        # Soften the probability distributions with temperature T
        # Higher T = Flatter distribution which reveals more relative information
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Calculate KL Divergence Loss
        loss_kl = self.kl_div_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 3. Combine Losses
        # Alpha balances the two losses
        total_loss = (self.alpha*loss_ce) + ((1-self.alpha)*loss_kl)
        return total_loss
