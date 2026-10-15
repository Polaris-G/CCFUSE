"""CCSFuse helpers: COCO weight transfer for the dual-stream YOLOv5."""
import logging

logger = logging.getLogger(__name__)


def load_coco_dual_stream(model, coco_model, pretrain_neck=False):
    """Copy single-stream COCO YOLOv5 weights into a dual-stream model.

    CFT's default transfer (intersect_dicts on identical keys) only fills the RGB backbone: the IR backbone lives
    at different layer indices, whose keys belong to the COCO neck. Here the model yaml's `pretrain_map` lists the
    dual-stream layer ids for COCO backbone layers 0..9 (rgb and ir) and, optionally, COCO neck layers 10..
    Returns (state_dict to load, {group: (n_loaded, n_total)}).
    """
    pmap = model.yaml.get('pretrain_map')
    assert pmap, 'model yaml has no pretrain_map; cannot map COCO weights onto the dual-stream model'
    src = coco_model.float().state_dict()
    dst = model.state_dict()
    groups = {'rgb': (pmap['rgb'], 0), 'ir': (pmap['ir'], 0)}
    if pretrain_neck:  # 推断 (论文未给出): paper only says the backbone is COCO-pretrained
        groups['neck'] = (pmap['neck'], len(pmap['rgb']))

    out, stats = {}, {}
    for g, (layers, src_offset) in groups.items():
        n_loaded = n_total = 0
        for pos, li in enumerate(layers):
            dprefix, sprefix = f'model.{li}.', f'model.{src_offset + pos}.'
            for k in dst:
                if not k.startswith(dprefix):
                    continue
                n_total += 1
                sk = sprefix + k[len(dprefix):]
                if sk in src and src[sk].shape == dst[k].shape:
                    out[k] = src[sk]
                    n_loaded += 1
        stats[g] = (n_loaded, n_total)
    return out, stats
