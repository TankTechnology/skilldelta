PYTHON ?= python
PLUGIN = plugins/dsh-skilldelta

.PHONY: help all demo test plugin-demo test-plugin package verify check-public
help:
	@echo "make demo          Predict skill benefit from synthetic history"
	@echo "make test          Run offline unit tests"
	@echo "make plugin-demo   Try the Harness router offline"
	@echo "make test-plugin   Test the Harness hook (npm ci first)"
	@echo "make package       Package only allowlisted public code"
	@echo "make check-public  Check the Git file inventory"

all: test demo plugin-demo
demo:
	$(PYTHON) examples/predict_before_execution.py
test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m unittest discover -s $(PLUGIN)/tests -p 'test_*.py' -v
plugin-demo:
	$(PYTHON) $(PLUGIN)/scripts/route.py route --support $(PLUGIN)/examples/support.json --query-index $(PLUGIN)/examples/queries.json --task-id demo-help --k 2
test-plugin:
	SKILLDELTA_TEST_PYTHON="$(PYTHON)" npm --prefix $(PLUGIN) test
package:
	$(PYTHON) scripts/package_release.py
verify:
	$(PYTHON) scripts/verify_manifest.py
check-public:
	$(PYTHON) scripts/public_release.py --tracked
