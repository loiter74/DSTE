import torch
from torch import nn

from model.inner.denoise_layer import DenoiseNetwork
from model.inner.deter_layer import Deterministic
from model.inner.diff_emb import DiffusionEmbedding
from model.inner.inference_layer import InferenceModel
from model.inner.observation_layer import ObservationModel
from model.inner.st_module.graph_layer import DynamicAgg, StaticAgg
from model.inner.st_module.net_module import Conv1d_with_init, MultiConv1d, GatedFusion, Attn_tem
from model.inner.st_module.time_layer import TemporalLearning



class DSTE(nn.Module): # 用于补全
    def __init__(self, config):
        super().__init__()

        self.deter = Deterministic(config['input_dim'], config['covariate_dim'], config['emd_channel'],
                                   config['tcn_channels'], config['tcn_kernel_size'], config['dropout'],
                                   )
        self.inference = InferenceModel(config['tcn_channels'], config['latent_channels'], config['num_latent_layers'])

        # diffusion step
        self.diffusion_embedding = nn.ModuleList([DiffusionEmbedding(config["num_steps"], config['tcn_channels'][i]) for i in  range(len(config['tcn_channels'])-1, -1, -1)])
        self.diffusion_projection = nn.Linear(config["diffusion_embedding_dim"], config['emd_channel'])
        nn.init.xavier_uniform_(self.diffusion_projection.weight)
        # aggreate
        self.dynamic_agg = nn.ModuleList(
            [DynamicAgg(pred_in=config['input_dim'], feat_in=config['covariate_dim'], channels=c) for c in
             reversed(config['tcn_channels'])])
        self.static_agg = nn.ModuleList(
            [StaticAgg(pred_in=config['input_dim'], channels=c) for c in reversed(config['tcn_channels'])])
        # attn
        self.forward_agg = nn.ModuleList([Attn_tem(heads=4, layers=2, channels=config['tcn_channels'][i]) for i in  range(len(config['tcn_channels'])-1, -1, -1)])
        # self.forward_time = nn.ModuleList([TemporalLearning(channels=c, nheads=4, is_cross=True) for c in reversed(tcn_channels)])

        # self.lstm = nn.LSTM(input_size=c, hidden_size=c, 2, batch_first=True)
        # self.fc = nn.Linear(hidden_size=c, output_size=c)

        self.observation = MultiConv1d(
            in_channels=sum(config["tcn_channels"]),
            out_channels=config['input_dim'],
            num_layers=3,
            channel_reduction='half',
            dropout=0.1
        )

    def forward(self,
                side_context, pred_context, side_target,
                noisy_data, A, context_missing, t
                ):


        # side_target = None #NO side
        d_c, d_t = self.deter(side_context, pred_context, side_target, noisy_data, A, context_missing)
        q_target, q_dists = self.inference(d_c, d_t, A[:, 0], context_missing)

        # q_target = d_t[::-1] # no DS
        agg_total = []
        # side_target = None # no DA
        if side_target is not None:
            for i, layers in enumerate(self.dynamic_agg):
                agg = layers(side_target, side_context, pred_context)
                agg_total += [agg]
        else:
            for i, layers in enumerate(self.static_agg):
                agg = layers(pred_context, A[:, 0])
                agg_total += [agg]


        for i, layers in enumerate(self.forward_agg):
            b, n, c, t = q_target[i].shape

            q_target[i] = q_target[i].reshape(b*n, c, t).permute(0, 2, 1) # b n c t
            agg_total[i] = agg_total[i].reshape(b*n, c, t).permute(0, 2, 1)
            q_target[i] = layers(q_target[i], q_target[i], agg_total[i]).permute(0, 2, 1)
            # q_target[i] = layers(q_target[i], base_shape, q_target[i]) # SA
            q_target[i] = q_target[i].reshape(b, n, c, t)

        p_y_pred = self.observation(q_target) # , diffusion_emb
        return p_y_pred