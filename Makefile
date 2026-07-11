# GCP parameters
GCP_ZONE ?= us-central1-b
VM_INSTANCE_NAME ?= instance-101
MACHINE_TYPE ?= e2-highmem-16
DISK_SIZE ?= 120GB

# Training parameters
DATASET_PATH ?= training
MODEL ?= encoder
EXPERIMENT_NAME ?=

gcp-create:
	@echo "Creating GCP instance..."
	gcloud compute instances create $(VM_INSTANCE_NAME) \
		--project amplified-hull-484821-b5 \
		--zone=$(GCP_ZONE) \
		--machine-type $(MACHINE_TYPE) \
		--boot-disk-size $(DISK_SIZE) 

gcp-delete:
	@echo "Deleting GCP instance..."
	gcloud compute instances delete $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-start:
	gcloud compute instances start $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-stop:
	gcloud compute instances stop $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-ssh:
	gcloud compute ssh $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-copy:
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/arc2/data .gcp_dump
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/arc2/fbc .gcp_dump
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/arc2/wandb .gcp_dump

gcp-send:
	gcloud compute scp --zone=$(GCP_ZONE) --recurse .gcp_dump $(VM_INSTANCE_NAME):/home/reidtaylor/arc2

train-model:
	@echo "Training the ARC Encoder..."
	uv run training.py

view-training:
	tensorboard --logdir logs/

clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models
	rm -rf ./logs
	rm -rf lightning_logs/

help:
	@echo "Available targets:"
	@echo "  train-model     	 - Train the model"
	@echo "  view-training       - View training logs via TensorBoard"
	@echo "  clean               - Clean up artifacts (models, logs)"
	@echo "  clean-tmp           - Clean up temporary artifacts only"
	@echo "  gcp-create          - Create a new GCP instance"
	@echo "  gcp-delete          - Delete a GCP instance"
	@echo "  gcp-start           - Start an existing GCP instance"
	@echo "  gcp-stop            - Stop an active GCP instance"
	@echo "  gcp-ssh             - SSH into an active GCP instance"
	@echo "  gcp-copy            - Copy models and logs from GCP instance"
	@echo "  gcp-copy-logs       - Copy only logs from GCP instance"
	@echo ""
	@echo "Configuration variables (can be overridden):"
	@echo "  GCP_ZONE=$(GCP_ZONE)"
	@echo "  VM_INSTANCE_NAME=$(VM_INSTANCE_NAME)"
	@echo "  MACHINE_TYPE=$(MACHINE_TYPE)"
	@echo "  DISK_SIZE=$(DISK_SIZE)"
	@echo ""

.PHONY: train-model view-training compile clean clean-tmp gcp-create gcp-delete gcp-start gcp-stop gcp-ssh gcp-copy gcp-copy-logs help