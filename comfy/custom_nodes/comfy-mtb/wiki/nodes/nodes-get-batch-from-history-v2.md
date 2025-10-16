Alternative approach to the original BatchFromHistory node.

    Instead of relying on the /history endpoint, this manages its own history
    in comfy's temp folder.

    Queue items without output are ignored in the count.
    