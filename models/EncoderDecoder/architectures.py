import torch

from models.IModel import IModel
from models.EncoderDecoder.custom_layers import Encoder, Decoder


class EncoderDecoder_v01(IModel):
    def __init__(self, args):
        super(EncoderDecoder_v01, self).__init__(args)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x: dict) -> torch.Tensor:
        assert 'inputs' in x,'There should be 1 input'
        assert len(x['inputs'].shape) == 4, 'input should be 4D tensor'

        feats = self.encoder(x['inputs'])
        out = self.decoder(feats)
        return out


# if __name__ == '__main__':
#    from dataloaders.dataloaderGetter import getOTSDataloaders
#    from utils.configParser import options
#    import torch
#
#    from loss.lossGetters import getMSELoss, getL1Loss
#
#    CONFIG_FILE_NAME = "../../configs/EncoderDecoder_v01.ini"
#    args = options(CONFIG_FILE_NAME)
#
#    dataloaders = getOTSDataloaders(args.argsDataset)
#    for k, v in dataloaders.items():
#        print(len(dataloaders[k]))
#    data = next(iter(dataloaders["train"]))
#    device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")
#    data["inputs"] = data["inputs"].to(device)
#    data["gts"] = data["gts"].to(device)
#
#    model = EncoderDecoder_v01(args.argsModel)
#    model.to(device)
#
#    """
#    import matplotlib.pyplot as plt
#    plt.imshow((data['inputs'].cpu().detach().numpy()[0].transpose(1, 2, 0)+1)/2)
#    plt.waitforbuttonpress()
#    """
#
#    out2 = model(data)
#    print(out2.shape)
#    final_result = (data["inputs"] + 1) / 2 - (out2 + 1) / 2
#    final_result[0] = final_result[0] - final_result[0].min()
#    final_result[0] = final_result[0] / final_result[0].max()
#
#    #import matplotlib.pyplot as plt
#    #plt.imshow(final_result.cpu().detach().numpy()[0].transpose(1, 2, 0))
#    #plt.waitforbuttonpress()
#
#    ## loss test
#
#    data['result'] = out2
#    loss_l1_func = getL1Loss(None)
#    loss_mse_func = getMSELoss(None)
#
#    loss_l1 = loss_l1_func(data)
#    loss_mse = loss_mse_func(data)
#
#    tmp = 0


