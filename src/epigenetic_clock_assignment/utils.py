from typing import Optional, Union, Literal
import logging
import os
import wget
import gzip
import zipfile
import shutil
import owncloud


def download_nextcloud_file(folder_url: str,
                            filename: str,
                            out_filepath: str,
                            password: str = ''):
    log = logging.getLogger('download_nextcloud_file')

    if os.path.exists(out_filepath):
        log.debug(f'file {out_filepath} already exists; do not download again')
        return

    out_dir = os.path.dirname(out_filepath)
    log.debug(f'creating directory "{out_dir}" if does not exists...')
    os.makedirs(out_dir, exist_ok=True)

    log.debug('downloading file "%s" from folder url "%s"', filename,
              folder_url)

    oc = owncloud.Client.from_public_link(folder_url, folder_password=password)
    oc.get_file(filename, local_file=out_filepath)

    log.debug('downloaded to "%s"', out_filepath)


def download_file(url: str,
                  out_filepath: str,
                  intermediate_archive_filepath: Optional[str] = None,
                  archive_format: Optional[Union[Literal['zip'],
                                                 Literal['gzip']]] = None):
    log = logging.getLogger('download_file')

    if not os.path.exists(out_filepath):
        log.debug(f'file {out_filepath} does not exists; need to download')

        out_dir = os.path.dirname(out_filepath)
        log.debug(f'creating directory "{out_dir}" if does not exists...')
        os.makedirs(out_dir, exist_ok=True)

        if intermediate_archive_filepath is not None:

            if os.path.exists(intermediate_archive_filepath):
                # delete before download
                os.remove(intermediate_archive_filepath)

            intermediate_archive_dir = os.path.dirname(
                intermediate_archive_filepath)
            os.makedirs(intermediate_archive_dir, exist_ok=True)

        log.debug(f'downloading...')
        wget.download(url=url,
                      out=intermediate_archive_filepath or out_filepath)

        if intermediate_archive_filepath is not None:
            log.debug(f'ungzipping {intermediate_archive_filepath}...')

            if archive_format == 'gzip':
                with gzip.open(intermediate_archive_filepath, 'rb') as infile:
                    with open(out_filepath, 'wb') as outfile:
                        shutil.copyfileobj(infile, outfile)
            elif archive_format == 'zip':
                log.debug(f'unzipping {intermediate_archive_filepath}...')

                with zipfile.ZipFile(intermediate_archive_filepath) as zf:
                    zf.extract(os.path.basename(out_filepath), out_dir)
            else:
                raise Exception(f'invalid archive format: {archive_format}')

            # delete intermediate file after gunzip
            os.remove(intermediate_archive_filepath)

        log.debug(f'file {out_filepath} ready')
    else:
        log.debug(f'file {out_filepath} already exists; do not download again')
