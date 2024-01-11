import torch
from torch import nn
from torch import optim
import config
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import RRDBNet, Discriminator
from tqdm import tqdm
from utils import gradient_penalty as gp
from dataset import get_dataloader
import warnings


warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_fn(
    train_dataloader,
    discriminator, generator,
    l1_loss_fn, percep_loss_fn,
    # d_scaler, g_scaler,
    d_optimizer, g_optimizer,
    epoch,
):
    loop = tqdm(train_dataloader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = generator(low_res)
            critic_real = discriminator(high_res)
            critic_fake = discriminator(fake.detach())
            gp = gradient_penalty(discriminator, high_res,
                                  fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )

        d_optimizer.zero_grad()
        # d_scaler.scale(loss_critic).backward()
        # d_scaler.step(d_optimizer)
        # d_scaler.update()
        loss_critic.backward()
        d_optimizer.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1_loss_fn(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(discriminator(fake))
            loss_for_vgg = percep_loss_fn(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        g_optimizer.zero_grad()
        # g_scaler.scale(gen_loss).backward()
        # g_scaler.step(g_optimizer)
        # g_scaler.update()
        gen_loss.backward()
        g_optimizer.step()

        # writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        # tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", generator)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )
        
    torch.save(generator.state_dict(),
                f"{config.CHECKPOINTS_PATH}superres_weights_{epoch}.pt")


def main(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    print("setting up")
    generator = RRDBNet()
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    generator.train()
    discriminator.train()

    train_dataloader = get_dataloader(batch_size)

    l1_loss_fn = nn.L1Loss()
    percep_loss_fn = VGGLoss()

    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    print("starting training")
    for epoch in range(epochs):
        train_fn(
            train_dataloader=train_dataloader,
            discriminator=discriminator,
            generator=generator,
            l1_loss_fn=l1_loss_fn,
            percep_loss_fn=percep_loss_fn,
            # d_scaler=d_scaler,
            # g_scaler=g_scaler,
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer
        )


if __name__ == "__main__":
    main()
