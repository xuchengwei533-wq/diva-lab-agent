# -*- coding: utf-8 -*-
"""
Create a Bailian (Model Studio) knowledge base (index) from a local PDF file.

Required env vars:
- ALIBABA_CLOUD_ACCESS_KEY_ID
- ALIBABA_CLOUD_ACCESS_KEY_SECRET
- WORKSPACE_ID

Optional env vars:
- BAILIAN_ENDPOINT (default: bailian.cn-beijing.aliyuncs.com)
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_bailian20231229.client import Client as BailianClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


# ========= Your fixed inputs =========
FILE_PATH = r"D:\xuchengwei\Rag文件夹\艺术与生活.pdf"
KB_NAME = "test_music"
# =====================================


def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def calculate_md5(file_path: Path) -> str:
    md5_hash = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_client() -> BailianClient:
    config = open_api_models.Config(
        access_key_id=require_env("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        access_key_secret=require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    )
    config.endpoint = os.environ.get("BAILIAN_ENDPOINT", "bailian.cn-beijing.aliyuncs.com")
    return BailianClient(config)


def apply_upload_lease(
    client: BailianClient,
    workspace_id: str,
    category_id: str,
    file_name: str,
    file_md5: str,
    file_size: int,
):
    headers = {}
    request = bailian_models.ApplyFileUploadLeaseRequest(
        file_name=file_name,
        md_5=file_md5,
        size_in_bytes=file_size,
    )
    runtime = util_models.RuntimeOptions()
    return client.apply_file_upload_lease_with_options(category_id, workspace_id, request, headers, runtime)


def upload_file_to_presigned_url(pre_signed_url: str, upload_headers: dict, file_path: Path) -> None:
    # Avoid loading whole file into memory; stream it.
    headers = {
        "X-bailian-extra": upload_headers.get("X-bailian-extra"),
        "Content-Type": upload_headers.get("Content-Type", "application/octet-stream"),
    }
    with file_path.open("rb") as f:
        resp = requests.put(pre_signed_url, data=f, headers=headers, timeout=300)
    resp.raise_for_status()


def add_file(
    client: BailianClient,
    workspace_id: str,
    lease_id: str,
    parser: str,
    category_id: str,
):
    headers = {}
    request = bailian_models.AddFileRequest(
        lease_id=lease_id,
        parser=parser,
        category_id=category_id,
    )
    runtime = util_models.RuntimeOptions()
    return client.add_file_with_options(workspace_id, request, headers, runtime)


def describe_file(client: BailianClient, workspace_id: str, file_id: str):
    headers = {}
    runtime = util_models.RuntimeOptions()
    return client.describe_file_with_options(workspace_id, file_id, headers, runtime)


def wait_for_file_parse_success(
    client: BailianClient,
    workspace_id: str,
    file_id: str,
    poll_interval_sec: int = 5,
    timeout_sec: int = 30 * 60,
) -> None:
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError("Timed out while waiting for file parse to complete.")

        resp = describe_file(client, workspace_id, file_id)
        status = getattr(resp.body.data, "status", None) or resp.body.data.status
        print(f"[DescribeFile] file_id={file_id}, status={status}")

        if status == "PARSE_SUCCESS":
            return
        if status == "PARSE_FAILED":
            msg = getattr(resp.body.data, "message", "") if resp.body.data else ""
            raise RuntimeError(f"File parse failed. status={status}, message={msg}")

        time.sleep(poll_interval_sec)


def create_index(
    client: BailianClient,
    workspace_id: str,
    file_id: str,
    name: str,
    structure_type: str = "unstructured",
    source_type: str = "DATA_CENTER_FILE",
    sink_type: str = "DEFAULT",
):
    headers = {}
    request = bailian_models.CreateIndexRequest(
        structure_type=structure_type,
        name=name,
        source_type=source_type,
        sink_type=sink_type,
        document_ids=[file_id],
    )
    runtime = util_models.RuntimeOptions()
    return client.create_index_with_options(workspace_id, request, headers, runtime)


def submit_index_job(client: BailianClient, workspace_id: str, index_id: str):
    headers = {}
    request = bailian_models.SubmitIndexJobRequest(index_id=index_id)
    runtime = util_models.RuntimeOptions()
    return client.submit_index_job_with_options(workspace_id, request, headers, runtime)


def get_index_job_status(client: BailianClient, workspace_id: str, index_id: str, job_id: str):
    headers = {}
    request = bailian_models.GetIndexJobStatusRequest(index_id=index_id, job_id=job_id)
    runtime = util_models.RuntimeOptions()
    return client.get_index_job_status_with_options(workspace_id, request, headers, runtime)


def wait_for_index_completed(
    client: BailianClient,
    workspace_id: str,
    index_id: str,
    job_id: str,
    poll_interval_sec: int = 5,
    timeout_sec: int = 60 * 60,
) -> None:
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError("Timed out while waiting for index job to complete.")

        resp = get_index_job_status(client, workspace_id, index_id, job_id)
        status = getattr(resp.body.data, "status", None) or resp.body.data.status
        print(f"[GetIndexJobStatus] index_id={index_id}, job_id={job_id}, status={status}")

        if status == "COMPLETED":
            return
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Index job failed. status={status}")

        time.sleep(poll_interval_sec)


def create_knowledge_base_from_local_file(file_path_str: str, kb_name: str) -> str:
    workspace_id = require_env("WORKSPACE_ID")
    file_path = Path(file_path_str)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    category_id = "default"           # default category (can be changed)
    parser = "DASHSCOPE_DOCMIND"      # default doc parser
    client = create_client()

    print("Step A: Compute file metadata...")
    file_name = file_path.name
    file_md5 = calculate_md5(file_path)
    file_size = file_path.stat().st_size

    print("Step B: Apply upload lease...")
    lease_resp = apply_upload_lease(client, workspace_id, category_id, file_name, file_md5, file_size)
    lease_id = lease_resp.body.data.file_upload_lease_id
    upload_url = lease_resp.body.data.param.url
    upload_headers = lease_resp.body.data.param.headers

    print("Step C: Upload file to presigned URL...")
    upload_file_to_presigned_url(upload_url, upload_headers, file_path)

    print("Step D: Add file to Bailian data center...")
    add_resp = add_file(client, workspace_id, lease_id, parser, category_id)
    file_id = add_resp.body.data.file_id
    print(f"Added file_id={file_id}")

    print("Step E: Wait for PARSE_SUCCESS...")
    wait_for_file_parse_success(client, workspace_id, file_id)

    print("Step F: Create index (knowledge base)...")
    index_resp = create_index(client, workspace_id, file_id, kb_name)
    index_id = index_resp.body.data.id
    print(f"Created index_id={index_id}")

    print("Step G: Submit index job...")
    submit_resp = submit_index_job(client, workspace_id, index_id)
    job_id = submit_resp.body.data.id
    print(f"Submitted index job_id={job_id}")

    print("Step H: Wait for index job COMPLETED...")
    wait_for_index_completed(client, workspace_id, index_id, job_id)

    print("Done.")
    return index_id


if __name__ == "__main__":
    index_id = create_knowledge_base_from_local_file(FILE_PATH, KB_NAME)
    print(f"\nSUCCESS. Knowledge base (index) id: {index_id}")
