#!/usr/bin/env python3
"""
List all available artifact versions for a specific run in Wandb.
"""

import argparse
import wandb
from settings import settings


def list_artifact_versions(project: str, run_id: str, artifact_type: str = "explanations"):
    """
    List all available artifact versions for a run.
    
    Args:
        project: Wandb project name (e.g., "raymondl/gpt2-small")
        run_id: Wandb run ID
        artifact_type: Type of artifacts to list (default: "explanations")
    """
    # Initialize Wandb API
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get the run
    try:
        run = api.run(f"{project}/{run_id}")
        print(f"Run: {run.name} (ID: {run_id})")
        print(f"Project: {project}")
        print(f"State: {run.state}")
        print("-" * 60)
    except wandb.errors.CommError:
        print(f"Error: Run {run_id} not found in project {project}")
        return
    
    # Get all artifacts from this run
    artifacts = list(run.logged_artifacts())
    
    # Filter by type if specified
    if artifact_type:
        artifacts = [a for a in artifacts if a.type == artifact_type]
    
    if not artifacts:
        print(f"No artifacts of type '{artifact_type}' found in this run")
        return
    
    # Group artifacts by name
    artifact_groups = {}
    for artifact in artifacts:
        base_name = artifact.name.rsplit(':', 1)[0]  # Remove version suffix
        if base_name not in artifact_groups:
            artifact_groups[base_name] = []
        artifact_groups[base_name].append(artifact)
    
    # Display artifacts and their versions
    for base_name, versions in artifact_groups.items():
        print(f"\nArtifact: {base_name}")
        print(f"Type: {versions[0].type}")
        print("Available versions:")
        
        # Sort by version
        versions.sort(key=lambda x: x.version)
        
        for artifact in versions:
            size_mb = artifact.size / (1024 * 1024) if hasattr(artifact, 'size') else 0
            print(f"  - {artifact.version}: {artifact.created_at} ({size_mb:.2f} MB)")
            
            # Show if this is the latest version
            if artifact == max(versions, key=lambda x: x.version):
                print(f"    ^ LATEST VERSION")


def main():
    parser = argparse.ArgumentParser(description="List artifact versions for a Wandb run")
    parser.add_argument("--project", type=str, required=True,
                        help="Wandb project (e.g., 'raymondl/gpt2-small')")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Wandb run ID")
    parser.add_argument("--type", type=str, default="explanations",
                        help="Artifact type to filter (default: explanations)")
    parser.add_argument("--all-types", action="store_true",
                        help="Show all artifact types")
    
    args = parser.parse_args()
    
    artifact_type = None if args.all_types else args.type
    list_artifact_versions(args.project, args.run_id, artifact_type)


if __name__ == "__main__":
    main() 