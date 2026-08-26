# ShopSimulator

This is the embedded ShopSimulator environment used by this repository. The
snapshot includes the product archive, Environment v2.4, Reward v4 and the
structured `/api/shop_agent` service. Its upstream source commit is recorded in
[`EMBEDDED_SOURCE.json`](EMBEDDED_SOURCE.json).

Users should not install or start this directory manually. From the repository
root, run:

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

Generated product JSON, search indexes, virtual environments and logs are not
committed.
