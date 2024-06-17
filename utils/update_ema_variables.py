def update_ema_variables(model, teacher_model, alpha, global_step):
    # 使用真實平均值直到指數平均值更準確
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for param_stud, param_teach in zip(model.parameters(), teacher_model.parameters()):
        param_teach.data.mul_(alpha).add_(1 - alpha, param_stud.data)