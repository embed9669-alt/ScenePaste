from scenepaste.sample_data import bundled_samples_root


def test_bundled_sample_resources_are_present():
    root = bundled_samples_root()
    assert (root / "objects" / "sample_person.json").is_file()
    assert (root / "backgrounds" / "sample_bg_sunny.jpg").is_file()
    assert (root / "templates" / "parameterized_mixed_traffic.json").is_file()
