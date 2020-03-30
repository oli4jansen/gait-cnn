import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets.samplers import RandomClipSampler, UniformClipSampler
from tqdm import tqdm

from dataset import preprocess_directory, VideoDataset, Kinetics400
from models import GaitNet
from torchvision.models.video import r2plus1d_18
import transforms as T


# def inference():
#     dataset_dir = 'tmp'
#
#     video = "/Users/o.f.jansen/Desktop/single_person_videos/violence/kick.mp4"
#     crops_dir = preprocess(video, dataset_dir)
#
#     image_dataset = ImageFolder(crops_dir)
#     dataloader = DataLoader(image_dataset, batch_size=1)
#
#     model = GaitNet()
#     model.eval()
#
#     output = None
#     for batch in tqdm(dataloader):
#         output = model(batch, hx=output)
#
#     print(output.size())
#

def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)


def train():
    dataset_dir = 'data/synth-cmu'
    clips_per_video = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Normalize according to Kinetics-400
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])

    transform_train = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((112, 112)),
        T.RandomHorizontalFlip(),
        normalize
    ])

    transform_test = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((112, 112)),
        normalize
    ])

    dataset_train = Kinetics400(
        dataset_dir,
        frames_per_clip=16,
        step_between_clips=2,
        transform=transform_train,
        frame_rate=15
    )

    exit()


    dataset_test = Kinetics400(
        dataset_dir,
        frames_per_clip=16,
        step_between_clips=2,
        transform=transform_test,
        frame_rate=15
    )

    train_sampler = RandomClipSampler(dataset_train.video_clips, clips_per_video)
    test_sampler = UniformClipSampler(dataset_test.video_clips, clips_per_video)

    train_data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=24,
        sampler=train_sampler, num_workers=2,
        pin_memory=True, collate_fn=collate_fn)

    test_data_loader = DataLoader(
        dataset_test, batch_size=24,
        sampler=test_sampler, num_workers=2,
        pin_memory=True, collate_fn=collate_fn)

    model = GaitNet()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(len(train_data_loader))

    for epoch in range(5):

        print('Epoch ' + str(epoch))

        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), 'checkpoint.pt')

    model.eval()

    # print(str(len(data_loader_test)) + ' videos')

    with torch.no_grad():
        for video, target in test_data_loader:
            print('-- Start inference --')
            print(video.size())
            output = model(video)
            print(output.size())
            print(output)
            # for clip_output in output:
            #     c = torch.argmax(clip_output)
            #     print(c)
            #     print(torch.max(clip_output))
            # print(torch.argmax(output))

    # print(output.size())

if __name__ == '__main__':
    train()
