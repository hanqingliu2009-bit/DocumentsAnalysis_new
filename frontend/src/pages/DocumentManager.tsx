import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  VStack,
  HStack,
  Badge,
  IconButton,
  useToast,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Spinner,
  Alert,
  AlertIcon,
} from '@chakra-ui/react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'
import { FiTrash2, FiDownload, FiRefreshCw } from 'react-icons/fi'
import { useDropzone } from 'react-dropzone'
import { format } from 'date-fns'

import {
  fetchDocuments,
  uploadDocument as uploadDocumentApi,
  deleteDocument as deleteDocumentApi,
} from '../services/api'

const uploadDocument = async (file: File): Promise<void> => {
  try {
    await uploadDocumentApi(file)
  } catch (e: unknown) {
    const detail =
      typeof e === 'object' && e !== null && 'response' in e
        ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
        : undefined
    throw new Error(
      typeof detail === 'string' ? detail : 'Failed to upload document',
    )
  }
}

const deleteDocument = async (id: string): Promise<void> => {
  try {
    await deleteDocumentApi(id)
  } catch {
    throw new Error('Failed to delete document')
  }
}

const FileDropzone = ({
  onUpload,
  isUploading,
}: {
  onUpload: (file: File) => void
  isUploading: boolean
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.forEach((file) => {
        onUpload(file)
      })
    },
    [onUpload],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: isUploading,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    multiple: true,
  })

  return (
    <Box
      {...getRootProps()}
      p={8}
      border="2px dashed"
      borderColor={isDragActive ? 'blue.400' : 'gray.300'}
      borderRadius="lg"
      bg={isDragActive ? 'blue.50' : 'gray.50'}
      cursor={isUploading ? 'wait' : 'pointer'}
      opacity={isUploading ? 0.85 : 1}
      transition="all 0.2s"
      _hover={
        isUploading
          ? undefined
          : { borderColor: 'blue.400', bg: 'blue.50' }
      }
    >
      <input {...getInputProps()} />
      <VStack spacing={3}>
        {isUploading ? (
          <>
            <Spinner size="lg" color="blue.500" />
            <Text fontSize="md" fontWeight="medium" color="gray.700" textAlign="center">
              Processing upload…
            </Text>
            <Text fontSize="sm" color="gray.500" textAlign="center" maxW="md">
              The server parses the file and builds embeddings. The first run can take several minutes
              while the embedding model downloads; keep this page open.
            </Text>
          </>
        ) : (
          <>
            <Text fontSize="lg" fontWeight="medium" color="gray.600">
              {isDragActive ? 'Drop files here...' : 'Drag & drop files here'}
            </Text>
            <Text fontSize="sm" color="gray.400">
              or click to select files
            </Text>
            <Text fontSize="xs" color="gray.400">
              Supported: PDF, DOCX, TXT
            </Text>
          </>
        )}
      </VStack>
    </Box>
  )
}

const StatusBadge = ({ status }: { status: string }) => {
  const statusColors: Record<string, string> = {
    completed: 'green',
    processing: 'yellow',
    pending: 'gray',
    failed: 'red',
  }

  return (
    <Badge colorScheme={statusColors[status] || 'gray'} textTransform="capitalize">
      {status}
    </Badge>
  )
}

const formatFileSize = (bytes?: number): string => {
  if (!bytes) return 'N/A'
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  return `${mb.toFixed(1)} MB`
}

const DocumentManager = () => {
  const toast = useToast()
  const queryClient = useQueryClient()

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    refetchInterval: 5000, // Refetch every 5 seconds to show processing status
  })

  const uploadMutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      toast({
        title: 'Document uploaded',
        status: 'success',
        duration: 3000,
      })
    },
    onError: (error: Error) => {
      toast({
        title: 'Upload failed',
        description: error.message,
        status: 'error',
        duration: 5000,
      })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      toast({
        title: 'Document deleted',
        status: 'success',
        duration: 3000,
      })
    },
    onError: (error: Error) => {
      toast({
        title: 'Delete failed',
        description: error.message,
        status: 'error',
        duration: 5000,
      })
    },
  })

  const handleUpload = (file: File) => {
    uploadMutation.mutate(file)
  }

  const handleDelete = (id: string) => {
    if (window.confirm('Are you sure you want to delete this document?')) {
      deleteMutation.mutate(id)
    }
  }

  const documents = data?.documents || []

  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Box>
          <Heading size="lg">Document Manager</Heading>
          <Text color="gray.500">Upload and manage your documents</Text>
        </Box>
        <Button
          leftIcon={<FiRefreshCw />}
          onClick={() => refetch()}
          isLoading={isLoading}
          variant="outline"
        >
          Refresh
        </Button>
      </Flex>

      <FileDropzone onUpload={handleUpload} isUploading={uploadMutation.isPending} />

      {isLoading && documents.length === 0 && (
        <Flex justify="center" py={12}>
          <Spinner size="xl" />
        </Flex>
      )}

      {error && (
        <Alert status="error" mt={6}>
          <AlertIcon />
          {error.message}
        </Alert>
      )}

      {documents.length > 0 && (
        <Box mt={8} bg="white" borderRadius="lg" boxShadow="sm" overflow="hidden">
          <Table variant="simple">
            <Thead bg="gray.50">
              <Tr>
                <Th>Title</Th>
                <Th>Type</Th>
                <Th>Status</Th>
                <Th>页数</Th>
                <Th>Chunks</Th>
                <Th>Size</Th>
                <Th>Uploaded</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {documents.map((doc) => (
                <Tr key={doc.id}>
                  <Td fontWeight="medium">{doc.title}</Td>
                  <Td>
                    <Badge variant="outline" textTransform="uppercase">
                      {doc.doc_type}
                    </Badge>
                  </Td>
                  <Td>
                    <StatusBadge status={doc.status} />
                  </Td>
                  <Td>
                    {typeof doc.metadata?.page_count === 'number'
                      ? doc.metadata.page_count
                      : '—'}
                  </Td>
                  <Td>{doc.chunk_count}</Td>
                  <Td>{formatFileSize(doc.metadata?.file_size)}</Td>
                  <Td>{format(new Date(doc.created_at), 'MMM d, yyyy')}</Td>
                  <Td>
                    <HStack spacing={2}>
                      <IconButton
                        aria-label="Download"
                        icon={<FiDownload />}
                        size="sm"
                        variant="ghost"
                        as="a"
                        href={`/api/documents/${doc.id}/download`}
                      />
                      <IconButton
                        aria-label="Delete"
                        icon={<FiTrash2 />}
                        size="sm"
                        variant="ghost"
                        colorScheme="red"
                        onClick={() => handleDelete(doc.id)}
                        isLoading={deleteMutation.isPending}
                      />
                    </HStack>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      )}

      {!isLoading && documents.length === 0 && !error && (
        <Box textAlign="center" py={12}>
          <Text color="gray.500">No documents uploaded yet.</Text>
          <Text color="gray.400" fontSize="sm">
            Use the dropzone above to upload your first document.
          </Text>
        </Box>
      )}
    </Box>
  )
}

export default DocumentManager
