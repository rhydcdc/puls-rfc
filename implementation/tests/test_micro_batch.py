from puls_sched.micro_batch import MicroBatch


def test_split_prefill_decode():
    mb = MicroBatch(
        id=0,
        layer_index=3,
        prefill_chunk={1: [10, 11, 12], 2: [20, 21]},
        decode_tokens={3: 1, 4: 1},
    )
    assert mb.prefill_chunk == {1: [10, 11, 12], 2: [20, 21]}
    assert mb.decode_tokens == {3: 1, 4: 1}
    assert mb.layer_index == 3
    assert not mb.is_pure_prefill()
    assert not mb.is_pure_decode()


def test_request_ids_union():
    mb = MicroBatch(
        id=0,
        prefill_chunk={1: [10], 2: [20]},
        decode_tokens={3: 1, 4: 1},
    )
    assert mb.request_ids() == {1, 2, 3, 4}

    mb_prefill_only = MicroBatch(id=1, prefill_chunk={5: [50]})
    assert mb_prefill_only.is_pure_prefill()
    assert mb_prefill_only.request_ids() == {5}

    mb_decode_only = MicroBatch(id=2, decode_tokens={6: 1})
    assert mb_decode_only.is_pure_decode()
    assert mb_decode_only.request_ids() == {6}


# =========================================================================
# Impl-5 — 신규 3 필드 (k_total · kv_rows_total · current_layer_index)
# =========================================================================

def test_micro_batch_default_k_total_zero():
    assert MicroBatch(id=0).k_total == 0


def test_micro_batch_default_kv_rows_total_zero():
    assert MicroBatch(id=0).kv_rows_total == 0


def test_micro_batch_default_current_layer_index_zero():
    assert MicroBatch(id=0).current_layer_index == 0


def test_micro_batch_explicit_field_roundtrip():
    mb = MicroBatch(id=0, k_total=2048, kv_rows_total=10000, current_layer_index=5)
    assert mb.k_total == 2048
    assert mb.kv_rows_total == 10000
    assert mb.current_layer_index == 5


def test_micro_batch_existing_methods_unchanged():
    """Impl-5 신규 필드가 기존 메서드 시맨틱 영향 0 (regression)."""
    mb = MicroBatch(
        id=0, k_total=512, kv_rows_total=999,
        prefill_chunk={1: [10]}, decode_tokens={2: 1},
    )
    assert mb.request_ids() == {1, 2}
    assert not mb.is_pure_prefill()
    assert not mb.is_pure_decode()


def test_micro_batch_layer_index_distinct_from_current():
    """layer_index (Impl-1, 시작 layer) vs current_layer_index (Impl-5, 현재 layer)
    의도된 의미 분리. 향후 통합은 Impl-9 driver wiring 시 검토 (§7 O5.1).

    *약한 lock-in (implementation hygiene 영역, ARCH 직접 정합 아님).*
    """
    fields = MicroBatch.__dataclass_fields__
    assert "layer_index" in fields
    assert "current_layer_index" in fields
    # 두 필드는 서로 다른 식별자 (alias 아님)
    mb = MicroBatch(id=0, layer_index=3, current_layer_index=7)
    assert mb.layer_index == 3
    assert mb.current_layer_index == 7
    assert mb.layer_index != mb.current_layer_index
