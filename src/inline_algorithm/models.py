'''
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

---

Defining Pydantic models
'''
from typing import List, Union
from pydantic import BaseModel, Field

class ScanStart(BaseModel):
    '''
    For the /scan/start API message
    '''
    algorithm_id: str
    slide_name: str
    stain_name: str
    organ_name: str
    tile_width: int
    tile_height: int
    path_to_output: str
    microns_per_pixel: dict | None = None
    available_magnifications: List[str] | None = None

class ScanOngoing(BaseModel):
    '''
    For the /scan/image-tile API message
    '''
    slide_name: str
    tile_name: str
    tile_image_path: str
    row_idx: int
    col_idx: int

class ScanEnd(BaseModel):
    '''
    For the /scan/end API message
    '''
    slide_name: str

class ScanAbort(BaseModel):
    '''
    For the /scan/abort API message
    '''
    slide_name: str

class DetectionArray(BaseModel):
    """
    Pydantic model
    """
    bbox: List[Union[int, float]]
    confidence: float
    class_: str = Field(..., alias='class')
    scan_at_other_mag: dict | None = None

    class Config:
        allow_population_by_field_name = True

class AoiResults(BaseModel):
    '''
    For the final results of an algorithm
    '''
    row_idx: int
    col_idx: int
    z_stack_to_preserve: bool | None = None

class TileResults(BaseModel):
    '''
    For the results per tile of an algorithm
    '''
    algorithm_id: str
    slide_name: str
    tile_name: str
    results: AoiResults
    scan_at_other_mag: dict | None = None
