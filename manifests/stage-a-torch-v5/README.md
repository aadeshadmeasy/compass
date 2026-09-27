# stage-a-torch-v5 (in progress)

**GPU train deferred until quota** for preferred `ml.g4dn.xlarge` training (Service Quotas request `ecee886646f74b26b6e725e723e812b49hRuW34F`, case `179054010300961`, status CASE_OPENED).

Also APPROVED (usable tomorrow for big GPU train): `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g4dn.2xlarge`.

Overnight GPU jobs were **stopped** per plan; big GPU train starts after g4dn.xlarge approval (or on approved g5/g4dn.2xlarge tomorrow).

Box train may produce an interim checkpoint; weights still gitignored.
