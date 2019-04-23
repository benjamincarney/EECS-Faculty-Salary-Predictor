from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from stem.process import launch_tor

tor_bin = r"C:\\Users\\Aaron\\Desktop\\Tor Browser\\Browser\\firefox.exe"
torrc_path = r"C:\\Users\\Aaron\\Desktop\\Tor Browser\\Browser\\TorBrowser\\Data\\Tor\\torrc"

tor_process = launch_tor(tor_cmd=tor_bin, torrc_path=torrc_path)

print('Running tor process...')

binary = FirefoxBinary(r"C:\\Users\\Aaron\\Desktop\\Tor Browser\\Browser\\firefox.exe")
profile = webdriver.FirefoxProfile()
profile.set_preference('network.proxy.type', 1)
profile.set_preference('network.proxy.socks', '127.0.0.1')
profile.set_preference('network.proxy.socks_port', 9150)
print(profile.profile_dir)
print(profile.userPrefs)
print(profile.default_preferences)
driver = webdriver.Firefox(profile)
print('Instantiated tor browser!')
print(type(driver))
print(type(driver.profile))

# # configured profile settings
# driver.profile.set_preference('network.proxy.type', 1)
# driver.profile.set_preference('network.proxy.socks', '127.0.0.1')
# driver.profile.set_preference('network.proxy.socks_port', 9051)

# driver.get("http://stackoverflow.com")
# driver.save_screenshot("screenshot.png")
# driver.quit()
