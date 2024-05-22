from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer


def get_model_decoder_layer(
    model_name: str,
) -> type[MixtralDecoderLayer]:
    if "Mixtral" in model_name:
        return MixtralDecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
