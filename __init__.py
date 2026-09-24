from comfy_api.latest import ComfyExtension

from . import optimizer_nodes
from .lossless import nodes as lossless_nodes


class NodePack(ComfyExtension):
    async def get_node_list(self):
        return optimizer_nodes.NODES + lossless_nodes.NODES


async def comfy_entrypoint() -> NodePack:
    return NodePack()
