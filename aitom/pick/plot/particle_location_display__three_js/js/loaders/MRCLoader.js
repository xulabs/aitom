THREE.MRCLoader = function (manager) {

	this.manager = (manager !== undefined) ? manager : THREE.DefaultLoadingManager;

};

Object.assign(THREE.MRCLoader.prototype, THREE.EventDispatcher.prototype, {

	load: function (url, onLoad, onProgress, onError) {
		var scope = this;

		var loader = new THREE.FileLoader(scope.manager);
		loader.setResponseType('arraybuffer');
		loader.load(url, function (data) {

			onLoad(scope.parse(data));

		}, onProgress, onError);

	},

	parse: function (data) {
		
		var _data = data;
		var _dataPointer = 0;
		var _nativeLittleEndian = new Int8Array(new Int16Array([1]).buffer)[0] > 0;
		var _littleEndian = true;

		function parseStream(_data) {

			var MRC = {
				nx: 0, //Number of Columns
				ny: 0, //Number of Rows
				nz: 0, //Number of Sections
				mode: 0, //Type of pixel in image. Values used by IMOD
				nxstart: 0, //Startin poin to sub image (not used in IMOD)
				nystart: 0,
				nzstart: 0,
				mx: 0, //Grid size in X, Y, Z
				my: 0,
				mz: 0,
				xlen: 0, //Cell size; pixel spacing = xlen/mx...
				ylen: 0,
				zlen: 0,
				alpha: 0, //cell angles - ignored by IMOD
				beta: 0,
				gamma: 0,
				mapc: 0, //map column
				mapr: 0, //map row
				maps: 0, //map section
				amin: 0, //Minimum pixel value
				amax: 0, //Maximum pixel value
				amean: 0, //mean pixel value
				ispg: 0, //space group numbe (ignored by IMOD0
				next: 0, //number of bytes in extended header
				creatid: 0, //is 0
				extra: null,
				nint: 0, // number of intergers or bytes per section
				nreal: 0, // Number of reals per section
				extra: null,
				imodStamp: 0, //1146047817 = file created by IMOD
				imodFlags: 0, //Bit flags
				idtype: 0,
				lens: 0,
				nd1: 0,
				nd2: 0,
				vd1: 0,
				vd2: 0,
				tiltangles: null,
				xorg: 0, //Orgin of the image
				yorg: 0,
				zorg: 0,
				cmap: 0, //Contains "MAP "
				stamp: 0, //Frist two bytes = 17 17 for bin-endian or 68 and 65 for littl-edian
				rms: 0, //RMS deviation of densitites from mean density
				nlabl: 0, //number of lables with useful data
				data: null, //10 lables of 80 characters
				min: Infinity,
				max: -Infinity,
				mean: 0,
				space: null,
				spaceorientation: null,
				rasspaceorientation: null,
				orientation: null,
				normcosine: null
			};

			_dataPointer = 0;		
			
			// Reading the data. Names are the names used in C code.
			MRC.nx = scan('sint'); console.log('nx = ' + MRC.nx);
			MRC.ny = scan('sint'); console.log('ny = ' + MRC.ny);
			MRC.nz = scan('sint'); console.log('nz = ' + MRC.nz);
			MRC.mode = scan('sint');

			_dataPointer = 28;

			MRC.mx = scan('sint'); console.log('mx = ' + MRC.mx);
			MRC.my = scan('sint'); console.log('my = ' + MRC.my);
			MRC.mz = scan('sint'); console.log('mz = ' + MRC.mz);

			// pixel spacing = xlem/mx
			MRC.xlen = scan('float'); console.log('xlen = ' + MRC.xlen);
			MRC.ylen = scan('float'); console.log('ylen = ' + MRC.ylen);
			MRC.zlen = scan('float'); console.log('zlen = ' + MRC.zlen);
			MRC.alpha = scan('float');
			MRC.beta = scan('float');
			MRC.gamma = scan('float');
			MRC.mapc = scan('sint');
			MRC.mapr = scan('sint');
			MRC.maps = scan('sint');
			MRC.amin = scan('float');
			MRC.amax = scan('float');
			MRC.amean = scan('float');
			MRC.ispeg = scan('sint');
			MRC.next = scan('sint');
			MRC.creatid = scan('short');

			//Not sure what to do with the extra data, says 30 for size
			MRC.nint = scan('short');
			MRC.nreal = scan('short');
			//Need to figure out extra data, 20 for size
			MRC.imodStamp = scan('sint');
			MRC.imodFLags = scan('sint');
			MRC.idtype = scan('short');
			MRC.lens = scan('short');
			MRC.nd1 = scan('short');
			MRC.nd2 = scan('short');
			MRC.vd1 = scan('short');
			MRC.vd2 = scan('short');

			// loop this around (6 different ones)
			MRC.tiltangles = scan('float', 6);

			_dataPointer = 196;

			MRC.xorg = scan('float');
			MRC.yorg = scan('float');
			MRC.zorg = scan('float');

			_dataPointer = 216;

			MRC.rms = scan('float');
			MRC.nlabl = scan('sint');

			// 10 of 80 characters
			MRC.lables = scan('schar', 10);

			// size of the image
			var volsize = MRC.nx * MRC.ny * MRC.nz;

			//Dealing with extended header
			//****************After the header you have all the data***********************************************//
			
			if (MRC.next != 0) {
				_dataPointer = MRC.next + 1024;
				
				switch (MRC.mode) {
				case 0:
					MRC.data = scan('schar', volsize);
					break;
				case 1:
					MRC.data = scan('sshort', volsize);
					break;
				case 2:
					MRC.data = scan('float', volsize);
					break;
				case 3:
					MRC.data = scan('uint', volsize);
					break;
				case 4:
					MRC.data = scan('double', volsize);
					break;
				case 6:
					MRC.data = scan('ushort', volsize);
					break;
				case 16:
					MRC.data = scan('uchar', volsize);
					break;

				default:
					throw new Error('Unsupported MRC data type: ' + MRC.mode);
				};
			}
			//****************After the header you have all the data***********************************************//

			// Read for the type of pixels --> Basically the mrc voxel data
			_dataPointer = 1024;
			
			console.log('MRC mode=' + MRC.mode);
			console.log('Total elements=' + volsize);
			
			switch (MRC.mode) {
			case 0:
				MRC.data = scan('schar', volsize);
				break;
			case 1:
				MRC.data = scan('sshort', volsize);
				break;
			case 2:
				MRC.data = scan('float', volsize);
				break;
			case 3:
				MRC.data = scan('uint', volsize);
				break;
			case 4:
				MRC.data = scan('double', volsize);
				break;
			case 6:
				MRC.data = scan('ushort', volsize);
				break;
			case 16:
				MRC.data = scan('uchar', volsize);
				break;

			default:
				throw new Error('Unsupported MRC data type: ' + MRC.mode);
			}

			// minimum, maximum, mean intensities
			// centered on the mean for best viewing ability
			if (MRC.amean - (MRC.amax - MRC.amean) < 0) {
				MRC.min = MRC.amin;
				MRC.max = MRC.amean + (MRC.amean - MRC.amin);
			} else {
				MRC.min = MRC.amean - (MRC.amax - MRC.amean);
				MRC.max = MRC.amax
			}

			return MRC;

		};

		function scan (type, chunks) {

			if (chunks === null || chunks === undefined) {

				chunks = 1;

			}

			var _chunkSize = 1;
			var _array_type = Uint8Array;

			switch (type) {

				// 1 byte data types
			case 'uchar':
				break;
			case 'schar':
				_array_type = Int8Array;
				break;
				// 2 byte data types
			case 'ushort':
				_array_type = Uint16Array;
				_chunkSize = 2;
				break;
			case 'sshort':
				_array_type = Int16Array;
				_chunkSize = 2;
				break;
				// 4 byte data types
			case 'uint':
				_array_type = Uint32Array;
				_chunkSize = 4;
				break;
			case 'sint':
				_array_type = Int32Array;
				_chunkSize = 4;
				break;
			case 'float':
				_array_type = Float32Array;
				_chunkSize = 4;
				break;
			case 'complex':
				_array_type = Float64Array;
				_chunkSize = 8;
				break;
			case 'double':
				_array_type = Float64Array;
				_chunkSize = 8;
				break;

			}
			
			var _start = _dataPointer;
			var _end = _dataPointer += chunks * _chunkSize;
			
			console.log(_start + ',' + _end);
			
			var _data_slice = _data.slice(_start, _end);
			var voxels = new _array_type(_data_slice);

			if (_nativeLittleEndian != _littleEndian) {
				voxels = flipEndianness(voxels, _chunkSize);
			}

			if (chunks == 1) {
				return voxels[0];
			}

			return voxels;

		};

		//Swapping the bits to match the endianness
		 function flipEndianness(array, chunkSize) {

			var u8 = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
			for (var i = 0; i < array.byteLength; i += chunkSize) {

				for (var j = i + chunkSize - 1, k = i; j > k; j--, k++) {

					var tmp = u8[k];
					u8[k] = u8[j];
					u8[j] = tmp;

				}

			}
			return array;
		};

		var MRC = parseStream(data);

		var volume = new THREE.Volume();

		// min and max intensities
		var min = MRC.min;
		var max = MRC.max;

		// attach the scalar range to the volume
		volume.windowLow = min;
		volume.windowHigh = max;

		//get dimsensions
		
		volume.xLength = MRC.nx;
		volume.yLength = MRC.ny;
		volume.zLength = MRC.nz;

		var _dimensions = [MRC.nx, MRC.ny, MRC.nz]; //voxel(i,j,k)
		volume.dimensions = _dimensions;

		//get voxel spacing
		var spacingX = MRC.xlen / MRC.mx;
		var spacingY = MRC.ylen / MRC.my;
		var spacingZ = MRC.zlen / MRC.mz;
		//volume.spacing = [spacingX, spacingY, spacingZ];
		volume.spacing = [1, 1, 1];

		// set the default threshold
		if (volume.lowerThreshold ===  - Infinity) {
			volume.lowerThreshold = min;
		}
		if (volume.upperThreshold === Infinity) {
			volume.upperThreshold = max;
		}

		//store the data into the volume
		_data = MRC.data;
		console.log('Data at last point' + _data[2686975]);

		volume.data = _data;

		// Create IJKtoRAS matrix
		volume.matrix = new THREE.Matrix4();

		//(Set these value as transpose of required matrix--three.js thing)
		volume.matrix.set(-1, 0, 0, 0,
			0, 0, -1, 0,
			0, -1, 0, 0,
			MRC.nx, MRC.ny, MRC.nz, 1);

		// Invert IJKtoRAS matrix
		volume.inverseMatrix = new THREE.Matrix4();
		volume.inverseMatrix.getInverse(volume.matrix);

		//Get RAS Dimensions
		volume.RASDimensions = (new THREE.Vector3(volume.xLength, volume.yLength, volume.zLength)).applyMatrix4(volume.matrix).round().toArray().map(Math.abs);

		/*
		// Transform ijk (0, 0, 0) to RAS
		var tar = goog.vec.Vec4.createFloat32FromValues(0, 0, 0, 1);
		var res = goog.vec.Vec4.createFloat32();
		goog.vec.Mat4.multVec4(IJKToRAS, tar, res);

		// Transform ijk (spacingX, spacinY, spacingZ) to RAS
		var tar2 = goog.vec.Vec4.createFloat32FromValues(1, 1, 1, 1);
		var res2 = goog.vec.Vec4.createFloat32();
		goog.vec.Mat4.multVec4(IJKToRAS, tar2, res2);

		// grab the RAS dimensions
		MRI.RASSpacing = [res2[0] - res[0], res2[1] - res[1], res2[2] - res[2]];
		MRI.RASDimensions = [_rasBB[1] + _rasBB[0] + 1, _rasBB[3] - _rasBB[2] + 1, _rasBB[5] - _rasBB[4] + 1];

		// grab the RAS Origin
		MRI.RASOrigin = [_rasBB[0], _rasBB[2], _rasBB[4]];
		 */
		return volume;

	} //end of parse

} //end of block of {load(),parse()}
); //end of Object.assign
