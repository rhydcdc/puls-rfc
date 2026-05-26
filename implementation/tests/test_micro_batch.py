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
