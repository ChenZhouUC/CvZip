from urllib.request import urlretrieve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def progressbar_urlretrieve(prefix='',
                            suffix='',
                            urlstr='',
                            decimals=1,
                            length=100,
                            finished='#',
                            pending=' ',
                            dynamic=True,
                            eps=1e-6):
    """
    hook function of urlretrieve to print a progress bar
    @params:
        prefix      - Optional  : prefix string (str)
        suffix      - Optional  : suffix string (str)
        urlstr      - Optional  : print after download (str)
        decimals    - Optional  : positive number of decimals in percent complete (int)
        length      - Optional  : character length of bar (int)
        finished    - Optional  : bar fill character (str)
        pending     - Optional  : bar pending character (str)
        dynamic     - Optional  : dynamic terminal interaction or not (bool)
        eps         - Optional  : progress precision control (float)
    """
    def progressbar_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        percent = ("{0:." + str(decimals) + "f}").format(progress * 100)
        filledLength = int(length * progress)
        bar = finished * filledLength + pending * (length - filledLength)
        if dynamic:
            if progress >= 1 - eps:
                endline = "\n" + urlstr + "\n"
            else:
                endline = "\r"
            print(f"\r{prefix} |{bar}| {percent}% [{suffix}]", end=endline)
        else:
            if progress >= 1 - eps:
                endline = "\n" + urlstr + "\n"
                print(f"\r{prefix} |{bar}| {percent}% [{suffix}]", end=endline)

    return progressbar_hook


if __name__ == "__main__":
    url = "https://whale-edge.oss-cn-shanghai.aliyuncs.com/openplatform/wop_ai_picture_library/cache/synopsis/2021-03-19-03-09-33.651096.mp4"
    download_path = "/home/chenzhou/Downloads/test.mp4"
    urlretrieve(url,
                download_path,
                reporthook=progressbar_urlretrieve(prefix='Progress:',
                                                   suffix='Downloaded',
                                                   urlstr=url + ' Downloaded',
                                                   length=100,
                                                   dynamic=False))
