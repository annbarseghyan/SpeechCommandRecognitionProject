import matplotlib.pyplot as plt


def plot_training_curves(train_loss_list, test_loss_list, train_acc_list, test_acc_list, num_epochs, save_path=None):
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_loss_list, label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_list, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_acc_list, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
