import os
import re
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

DATA_CACHE_FILE = 'historical_cache.json'

class PCSODataScraper:
    """
    Scraper and data manager for Philippine PCSO Ultra Lotto 6/58.
    Supports headless browser extraction, local JSON cache persistence, and fallback generation.
    """
    def __init__(self, cache_file=DATA_CACHE_FILE):
        self.search_url = "https://www.pcso.gov.ph/SearchLottoResult.aspx"
        self.cache_file = cache_file

    def get_latest_winning_numbers(self):
        """Fetch the latest draw from PCSO or return recent cached/sample draw."""
        results = self._fetch_pcso_draws(days_back=14)
        if results:
            return results[0]
        cached = self.load_data_cache()
        if cached and cached.get('latest_numbers'):
            return {
                'numbers': cached['latest_numbers'],
                'date': cached['latest_date'],
                'jackpot': cached['latest_jackpot'],
                'winners': cached['latest_winners'],
                'is_sample': False
            }
        return self._make_sample_result()

    def scrape_historical_data(self, days_back=365):
        """Scrape full historical draws from PCSO; fall back to cache or sample."""
        results = self._fetch_pcso_draws(days_back=days_back)
        if results:
            return results
        cached = self.load_data_cache()
        if cached and cached.get('historical_data'):
            return cached['historical_data']
        return self.generate_sample_data()

    def generate_sample_data(self, periods=180):
        """Generate realistic synthetic 6/58 draws when offline or during initial startup."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        # Ultra lotto 6/58 draws are on Tue, Fri, Sun
        lotto_dates = [d.to_pydatetime() for d in dates if d.weekday() in [1, 4, 6]]
        if not lotto_dates:
            lotto_dates = [dates[-1].to_pydatetime()]
            
        draws = []
        for dt in reversed(lotto_dates):
            nums = sorted(np.random.choice(range(1, 59), 6, replace=False).tolist())
            draws.append({
                'numbers': nums,
                'date': dt,
                'jackpot': f"₱{np.random.randint(49_500_000, 350_000_000):,.2f}",
                'winners': 1 if np.random.random() < 0.05 else 0,
                'is_sample': True
            })
        return draws

    def _fetch_pcso_draws(self, days_back=180):
        """Headless Chrome automated search on PCSO portal."""
        if not SELENIUM_AVAILABLE:
            return []

        driver = None
        try:
            options = Options()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument(
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
            )

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            driver.get(self.search_url)
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'select')))

            today = datetime.now()
            start = today - timedelta(days=days_back)

            self._select_by_value_or_text(driver, 'ddlStartMonth', start.strftime('%B'))
            self._select_by_value_or_text(driver, 'ddlStartDate', str(start.day))
            self._select_by_value_or_text(driver, 'ddlStartYear', str(start.year))

            self._select_by_value_or_text(driver, 'ddlEndMonth', today.strftime('%B'))
            self._select_by_value_or_text(driver, 'ddlEndDay', str(today.day))
            self._select_by_value_or_text(driver, 'ddlEndYear', str(today.year))

            # Select 6/58 Ultra Lotto game
            selects = driver.find_elements(By.TAG_NAME, 'select')
            for sel in selects:
                sel_id = sel.get_attribute('id') or ''
                sel_name = sel.get_attribute('name') or ''
                if 'game' in sel_id.lower() or 'game' in sel_name.lower():
                    s = Select(sel)
                    for opt in s.options:
                        if '6/58' in opt.text.lower() or 'ultra lotto' in opt.text.lower():
                            s.select_by_visible_text(opt.text)
                            break
                    break

            # Click search
            buttons = driver.find_elements(By.TAG_NAME, 'input')
            for btn in buttons:
                if btn.get_attribute('type') == 'submit' or 'search' in (btn.get_attribute('value') or '').lower():
                    btn.click()
                    break

            time.sleep(3)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            return self._parse_all_rows(soup)
        except Exception:
            return []
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def _select_by_value_or_text(self, driver, field_hint, value):
        selects = driver.find_elements(By.TAG_NAME, 'select')
        for sel in selects:
            sel_id = (sel.get_attribute('id') or '').lower()
            sel_name = (sel.get_attribute('name') or '').lower()
            if field_hint.lower() in sel_id or field_hint.lower() in sel_name:
                s = Select(sel)
                try:
                    s.select_by_value(value)
                    return
                except Exception:
                    pass
                for opt in s.options:
                    if opt.text.strip().lower() == str(value).strip().lower():
                        s.select_by_visible_text(opt.text)
                        return

    def _parse_all_rows(self, soup):
        draws = []
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                cells = [c.get_text(strip=True) for c in row.find_all('td')]
                if len(cells) < 4:
                    continue
                game_cell = cells[0].strip()
                if not re.search(r'6/58', game_cell, re.IGNORECASE):
                    continue

                combination = cells[1].strip()
                draw_date = cells[2].strip()
                jackpot_raw = cells[3].strip()
                winners_raw = cells[4].strip() if len(cells) > 4 else '0'

                nums_match = re.findall(r'(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})', combination)
                if not nums_match:
                    continue
                numbers = sorted([int(n) for n in nums_match[0]])

                draw_dt = self._parse_date(draw_date)
                jackpot = self._parse_jackpot(jackpot_raw)
                try:
                    winners = int(winners_raw)
                except ValueError:
                    winners = 0

                draws.append({
                    'numbers': numbers,
                    'date': draw_dt,
                    'jackpot': jackpot,
                    'winners': winners,
                    'is_sample': False
                })

        return draws[::-1]  # Newest first

    def _parse_date(self, text):
        for fmt in ['%m/%d/%Y', '%B %d, %Y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return datetime.now()

    def _parse_jackpot(self, text):
        cleaned = re.sub(r'[^\d,.]', '', text).strip()
        if cleaned:
            try:
                amount = float(cleaned.replace(',', ''))
                return f"₱{amount:,.2f}"
            except ValueError:
                pass
        return "₱0.00"

    def _make_sample_result(self):
        return {
            'numbers': sorted(np.random.choice(range(1, 59), 6, replace=False).tolist()),
            'date': datetime.now(),
            'jackpot': '₱49,500,000.00',
            'winners': 0,
            'is_sample': True
        }

    def save_data_cache(self, historical_data, data_source, latest_numbers, latest_date, latest_jackpot, latest_winners):
        """Save data to local JSON cache."""
        try:
            cache = {
                'historical_data': [
                    {
                        'numbers': d['numbers'],
                        'date': d['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(d['date'], 'strftime') else str(d['date']),
                        'jackpot': d.get('jackpot', ''),
                        'winners': d.get('winners', 0),
                        'is_sample': d.get('is_sample', False)
                    }
                    for d in historical_data
                ],
                'data_source': data_source,
                'latest_numbers': latest_numbers,
                'latest_date': latest_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest_date, 'strftime') else str(latest_date),
                'latest_jackpot': latest_jackpot,
                'latest_winners': latest_winners
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_data_cache(self):
        """Load data from local JSON cache."""
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            historical_data = []
            for d in cache.get('historical_data', []):
                dt_raw = d['date']
                try:
                    dt = datetime.strptime(dt_raw, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    try:
                        dt = datetime.strptime(dt_raw[:10], '%Y-%m-%d')
                    except Exception:
                        dt = datetime.now()
                historical_data.append({
                    'numbers': d['numbers'],
                    'date': dt,
                    'jackpot': d.get('jackpot', '₱0.00'),
                    'winners': d.get('winners', 0),
                    'is_sample': d.get('is_sample', False)
                })
            if not historical_data:
                return None
            latest_date_raw = cache.get('latest_date', '')
            try:
                latest_date = datetime.strptime(latest_date_raw, '%Y-%m-%d %H:%M:%S')
            except Exception:
                latest_date = datetime.now()

            return {
                'historical_data': historical_data,
                'data_source': cache.get('data_source', 'live'),
                'latest_numbers': cache.get('latest_numbers', []),
                'latest_date': latest_date,
                'latest_jackpot': cache.get('latest_jackpot', '₱0.00'),
                'latest_winners': cache.get('latest_winners', 0)
            }
        except Exception:
            return None
