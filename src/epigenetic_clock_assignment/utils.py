from typing import Optional
import logging
import os
import wget
import gzip
import shutil


def download_file(url: str,
                  out_filepath: str,
                  intermediate_gzip_filepath: Optional[str] = None):
    log = logging.getLogger('download_file')

    if not os.path.exists(out_filepath):
        log.debug(f'file {out_filepath} does not exists; downloading...')

        if intermediate_gzip_filepath is not None and os.path.exists(
                intermediate_gzip_filepath):
            # delete before download
            os.remove(intermediate_gzip_filepath)

        wget.download(url=url, out=intermediate_gzip_filepath or out_filepath)

        if intermediate_gzip_filepath is not None:
            log.debug(f'unzipping {intermediate_gzip_filepath}...')

            with gzip.open(intermediate_gzip_filepath, 'rb') as infile:
                with open(out_filepath, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)

            # delete intermediate file after gunzip
            os.remove(intermediate_gzip_filepath)

        log.debug(f'file {out_filepath} ready')
    else:
        log.debug(f'file {out_filepath} already exists; do not download again')
