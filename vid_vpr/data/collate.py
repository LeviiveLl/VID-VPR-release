import torch


def pad_vlm_sequences(hidden_states, attention_masks):
    max_length = max(item.shape[0] for item in hidden_states)
    hidden_dim = hidden_states[0].shape[-1]
    padded_hidden = []
    padded_masks = []
    for hidden, mask in zip(hidden_states, attention_masks):
        pad_length = max_length - hidden.shape[0]
        if pad_length:
            hidden = torch.cat(
                [
                    hidden,
                    torch.zeros(
                        pad_length,
                        hidden_dim,
                        dtype=hidden.dtype,
                    ),
                ]
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(pad_length, dtype=mask.dtype),
                ]
            )
        padded_hidden.append(hidden)
        padded_masks.append(mask)
    return torch.stack(padded_hidden), torch.stack(padded_masks)


def train_collate(batch):
    if len(batch[0]) == 2:
        images, place_ids = zip(*batch)
        return torch.stack(images), torch.stack(place_ids)
    images, place_ids, hidden_nested, mask_nested = zip(*batch)
    hidden = [item for group in hidden_nested for item in group]
    masks = [item for group in mask_nested for item in group]
    padded_hidden, padded_masks = pad_vlm_sequences(hidden, masks)
    return (
        torch.stack(images),
        torch.stack(place_ids),
        padded_hidden,
        padded_masks,
    )


def evaluation_collate(batch):
    if len(batch[0]) == 2:
        images, indices = zip(*batch)
        return torch.stack(images), torch.tensor(indices)
    images, indices, hidden, masks = zip(*batch)
    padded_hidden, padded_masks = pad_vlm_sequences(list(hidden), list(masks))
    return (
        torch.stack(images),
        torch.tensor(indices),
        padded_hidden,
        padded_masks,
    )
