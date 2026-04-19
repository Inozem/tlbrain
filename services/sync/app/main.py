from services.sync.app.runner import run_sync


def sync_entry(request=None):
    run_sync()
    return "ok"
