import jax


@jax.tree_util.Partial
def loss(y_true, y_pred):
    return (y_pred - y_true["pred"]) ** 2


@jax.tree_util.Partial
def metrics(y_true, y_pred):
    center = ((y_pred - y_true["pred"]) ** 2)[:, -6:].mean()
    return {"center": center}