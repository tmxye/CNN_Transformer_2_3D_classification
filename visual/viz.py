from matplotlib import pyplot as plt


def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        content = [eval(line.replace("\n", "")) for line in file.readlines()]
        return content

def get_train_data(data):
    assert isinstance(data, list), "``data`` should be list type"
    epoch = []
    acc = []
    loss = []
    for line in data:
        if line["mode"] == "train":
            epoch.append(line["epoch"])
            acc.append(line["accuracy"])
            loss.append(line["loss"])
    return {"epoch": epoch, "accuracy": acc, "loss": loss}

def get_val_data(data):
    assert isinstance(data, list), "``data`` should be ``list`` type"
    epoch = []
    acc = []
    loss = []
    for line in data:
        if line["mode"] == "test":
            epoch.append(line["epoch"])
            acc.append(line["accuracy"])
            loss.append(line["loss"])
    return {"epoch": epoch, "accuracy": acc, "loss": loss}    

def get_best_val_acc(data):
    assert isinstance(data, list), "``data`` must be ``list`` type"
    return max(data)

def show_train(data):
    assert isinstance(data, dict), "``data`` should be ``dict`` type"
    _, axs = plt.subplots(1, 2,figsize=(10, 4), sharey=False)
    axs[0].plot(data["epoch"], data["accuracy"], label="train_acc", linewidth=1.5)
    axs[0].set_title("accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("train_acc")

    axs[1].plot(data["epoch"], data["loss"], label="train_loss", linewidth=1.5)
    axs[1].set_title("loss")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("train_loss")
    # plt.show()
    plt.savefig('../visual/01train_data.jpg', dpi=300)
    plt.clf()
def show_val(data):
    assert isinstance(data, dict), "``data`` should be ``dict`` type"
    _, axs = plt.subplots(1, 2,figsize=(10, 4), sharey=False)
    axs[0].plot(data["epoch"], data["accuracy"], label="val_acc", linewidth=1.5)
    axs[0].set_title("accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("val_acc")

    axs[1].plot(data["epoch"], data["loss"], label="val_loss", linewidth=1.5)
    axs[1].set_title("loss")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("val_loss")
    # plt.show()
    plt.savefig('../visual/01val_data.jpg', dpi=300)
    plt.clf()


def show_acc_loss(train_data, val_data):
    assert isinstance(train_data, dict), "``data`` should be ``dict`` type"
    assert isinstance(val_data, dict), "``data`` should be ``dict`` type"
    _, axs = plt.subplots(1, 2,figsize=(10, 4), sharey=False)
    axs[0].plot(train_data["epoch"], train_data["accuracy"], label="train_acc", linewidth=1.5)
    axs[0].plot(val_data["epoch"], val_data["accuracy"], label="val_acc", linewidth=1.5)
    axs[0].set_title("accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("acc")

    axs[1].plot(train_data["epoch"], train_data["loss"], label="train_loss", linewidth=1.5)
    axs[1].plot(val_data["epoch"], val_data["loss"], label="val_loss", linewidth=1.5)
    axs[1].set_title("loss")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("loss")
    # plt.show()
    plt.savefig('../visual/01acc_loss.jpg', dpi=300)
    plt.clf()

if __name__ == "__main__":
    # path = "F:/mixed-densenet/state/state_full_in_loop.txt"
    # path = "../state/state_full_in_loop.txt"
    path = "../visual/state1.txt"
    data = read_txt(path)
    train_data = get_train_data(data)
    print("best train accuracy:")
    print(get_best_val_acc(train_data["accuracy"]))
    val_data = get_val_data(data)
    print("best val accuracy:")
    print(get_best_val_acc(val_data["accuracy"]))
    show_train(train_data)
    show_val(val_data)
    show_acc_loss(train_data, val_data)
