from NSM.models import TriplanarDecoder
import torch


def test_triplanar(n_pts=1000, latent_size=512, deep_size=2, n_objects=2):
    # create a latent
    latent = torch.zeros(n_pts, latent_size)
    latent[: n_pts // 2, :] = torch.rand(latent_size)
    latent[n_pts // 2 :, :] = torch.rand(latent_size)
    # create theoretic xyz points
    xyz = (torch.rand(n_pts, 3) * 2) - 1

    # create a triplanar decoder
    triplanar = TriplanarDecoder(
        latent_dim=latent_size,
        n_objects=n_objects,
        conv_deep_image_size=deep_size,
        conv_pred_sdf=True,
    )

    # setup optimizer for triplanar
    optimizer = torch.optim.Adam(triplanar.parameters(), lr=1e-3)

    # train triplanar
    triplanar.train()
    sdf = triplanar(torch.cat([latent, xyz], dim=1))

    assert sdf.shape[0] == n_pts
    assert sdf.shape[1] == n_objects

    # calculate loss & perform backprop
    loss = sdf.sum()
    loss.backward()
    optimizer.step()

    sdf = triplanar(torch.cat([latent, xyz], dim=1))
    assert sdf.sum() < loss


if __name__ == "__main__":
    test_triplanar()
