import torch


def train(model, train_loader, test_loader, criterion, optimizer, checkpoint_path, device, num_epochs, scheduler=None, cnn=True):
    best_accuracy = 0.0
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for mfccs, labels in train_loader:
            mfccs = mfccs.to(torch.float32)

            labels = labels.to(device)
            mfccs = mfccs.to(device)

            optimizer.zero_grad()

            if cnn:
                mfccs = mfccs.unsqueeze(1)
            else:
                mfccs = torch.flatten(mfccs, start_dim=1)

            outputs = model(mfccs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for mfccs, labels in test_loader:
                mfccs = mfccs.to(torch.float32).to(device)
                labels = labels.to(device)

                if cnn:
                    mfccs = mfccs.unsqueeze(1)
                else:
                    mfccs = torch.flatten(mfccs, start_dim=1)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}]\n"
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%\n"
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\n"
              )

        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_test_loss)

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        if scheduler is not None:
            scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list
