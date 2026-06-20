import torch
import torch.nn as nn

# >>> x
# tensor([0.1846, 0.0393, 0.1422, 0.8559])
x = torch.rand(4)
# >>> l1.weight
# Parameter containing:
# tensor([[-0.1056, -0.0517, -0.1628, -0.2149],
#         [ 0.4311,  0.0334, -0.1531,  0.3269],
#         [ 0.4953,  0.0528,  0.2346,  0.3939]], requires_grad=True)
# >>> l1.bias
# Parameter containing:
# tensor([ 0.0849,  0.0985, -0.0511], requires_grad=True)
l1 = nn.Linear(4,3)
# >>> l2.weight
# Parameter containing:
# tensor([[-0.3798,  0.3847,  0.3913],
#         [ 0.0095,  0.2447, -0.0443],
#         [ 0.1018,  0.4449,  0.5577],
#         [ 0.2089,  0.3409,  0.2277]], requires_grad=True)
# >>> l2.bias
# Parameter containing:
# tensor([-0.5507, -0.4559, -0.4941, -0.3273], requires_grad=True)
l2 = nn.Linear(3,4)
# >>> z1
# tensor([-0.1437,  0.4374,  0.4130], grad_fn=<ViewBackward0>)
z1 = l1(x)
# >>> a1
# tensor([0.4641, 0.6076, 0.6018], grad_fn=<SigmoidBackward0>)
a1 = torch.sigmoid(z1)

# >>> z2
# tensor([-0.2578, -0.3295,  0.1592,  0.1138], grad_fn=<ViewBackward0>)
z2 = l2(a1)

# >>> loss
# tensor(-0.3142, grad_fn=<SumBackward0>)
loss = z2.sum()
loss.backward()


# 手算梯度

# loss self gradient
# >>> loss_self_grad
# tensor(1.)
loss_self_grad = torch.ones_like(loss)

# loss to z2 gradient
# >>> loss_z2_grad
# tensor([1., 1., 1., 1.])
loss_z2_grad = loss.grad_fn(loss_self_grad)

# gradient of z2 to a1, first: view, then: addmm
# view:
# >>> z2_view_grad
# tensor([[1., 1., 1., 1.]])
z2_view_grad = z2.grad_fn(loss_z2_grad)

# AddmmBackward0
# >>> with torch.no_grad():
# ...     z2_addmm_grad = z2.grad_fn.next_functions[0][0](z2_view_grad)
# ... 
# >>> z2_addmm_grad
# (tensor([[1., 1., 1., 1.]]), tensor([[-0.0596,  1.4152,  1.1324]]), tensor([[0.4641, 0.4641, 0.4641, 0.4641],
#         [0.6076, 0.6076, 0.6076, 0.6076],
#         [0.6018, 0.6018, 0.6018, 0.6018]]))

# z2_addmm_grad output is as top: 
# 1. first is gradient of l2.bias which is stored in l2.bias.grad
# 2. second is gradient of a1. however because a1 is not a leaf node, so this grad is just passed to a1.grad_fn
# 3. third is gradient of l2.weight which is stored in l2.weight.grad

# should be run under no_grad, or output tensors will have grad_fn which is not needed in such a scene
with torch.no_grad():
    z2_addmm_grad = z2.grad_fn.next_functions[0][0](z2_view_grad)

# so with z2_addmm_grad, we will handle the three output tensors
# 1. first tensor is stored in l2.bias.grad
l2.bias.grad = z2_addmm_grad[0]
# 2. third tensor is stored in l2.weight.grad
l2.weight.grad = z2_addmm_grad[2]
# 3. second tensor is passed to a1.grad_fn
with torch.no_grad():
    # >>> a1_to_z1_grad
    # tensor([[-0.0148,  0.3374,  0.2714]])
    a1_to_z1_grad = a1.grad_fn(z2_addmm_grad[1])

# pass a1_to_z1_grad to z1.grad_fn
# >>> z1_view_grad
# tensor([[-0.0148,  0.3374,  0.2714]])
z1_view_grad = z1.grad_fn(a1_to_z1_grad)
# continue to z1.grad_fn.next_functions[0][0]
# >>> z1_addmm_grad
# (tensor([[-0.0148,  0.3374,  0.2714]]), None, tensor([[-0.0027,  0.0623,  0.0501],
#         [-0.0006,  0.0132,  0.0107],
#         [-0.0021,  0.0480,  0.0386],
#         [-0.0127,  0.2888,  0.2323]]))
z1_addmm_grad = z1.grad_fn.next_functions[0][0](z1_view_grad)

# again, we should process the three output tuple elements like z2_addmm_grad
# 1. first tensor is stored in l1.bias.grad
l1.bias.grad = z1_addmm_grad[0]
# 2. third tensor is stored in l1.weight.grad
l1.weight.grad = z1_addmm_grad[2]
# 3. second tensor is discarded since x has no requires_grad and no grad_fn property