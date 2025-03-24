# Copyright 2021-2023 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda.ccuda as ccuda
from cuda.ccudart cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool

cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil
cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef const char* _cudaGetErrorString(cudaError_t error) nogil except ?NULL
cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDevice(int* device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSetDevice(int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaPitchedPtr _make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil
cdef cudaPos _make_cudaPos(size_t x, size_t y, size_t z) nogil
cdef cudaExtent _make_cudaExtent(size_t w, size_t h, size_t d) nogil
cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaIpcCloseMemHandle(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetPriority(cudaStream_t hStream, int* priority) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaCtxResetPersistingL2Cache() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamDestroy(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamSynchronize(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamQuery(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventSynchronize(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventDestroy(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDestroyExternalMemory(cudaExternalMemory_t extMem) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMalloc(void** devPtr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFree(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFreeHost(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFreeArray(cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaHostAlloc(void** pHost, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaHostRegister(void* ptr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaHostUnregister(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaHostGetFlags(unsigned int* pFlags, void* pHost) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemGetInfo(size_t* free, size_t* total) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset(void* devPtr, int value, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFreeAsync(void* devPtr, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolDestroy(cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDestroyTextureObject(cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGraphMemTrim(int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphDestroyNode(cudaGraphNode_t node) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t graphExec) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphDestroy(cudaGraph_t graph) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaProfilerStart() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaProfilerStop() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) nogil except ?cudaErrorCallRequiresNewerDriver
