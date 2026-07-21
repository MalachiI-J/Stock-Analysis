from pathlib import Path

from stock_scrapper.config import load_config, load_watchlist


def test_config_loading_reads_settings_and_watchlist(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    settings_path = config_dir / "settings.yaml"
    watchlist_path = config_dir / "watchlist.csv"
    settings_path.write_text("app_name: TestApp\nwatchlist_path: config/watchlist.csv\ndatabase_path: data/market.db\n", encoding="utf-8")
    watchlist_path.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")

    config = load_config(base_dir=tmp_path)
    watchlist = load_watchlist(watchlist_path)

    assert config["app_name"] == "TestApp"
    assert watchlist == ["AAPL", "MSFT"]
