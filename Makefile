.PHONY: run run-container gcloud-deploy

run:
	streamlit run Apps.py --server.port=8080 --server.address=0.0.0.0
