import { Box, Button, Flex, Heading, SimpleGrid, Stat, StatHelpText, StatLabel, StatNumber, Text, VStack } from '@chakra-ui/react'
import { useQuery } from '@tanstack/react-query'
import { FiFileText, FiLayers, FiMessageCircle } from 'react-icons/fi'
import { Link } from 'react-router-dom'

import { fetchStats } from '../services/api'

const StatCard = ({ title, value, icon: Icon, link, helpText }: { title: string; value: number | string; icon: any; link?: string; helpText?: string }) => {
  const content = (
    <Stat
      p={6}
      bg="white"
      borderRadius="lg"
      boxShadow="sm"
      borderWidth="1px"
      borderColor="gray.100"
      _hover={link ? { boxShadow: 'md', borderColor: 'blue.200' } : undefined}
      transition="all 0.2s"
    >
      <Flex justify="space-between" align="flex-start">
        <Box>
          <StatLabel fontSize="sm" color="gray.500" fontWeight="medium">
            {title}
          </StatLabel>
          <StatNumber fontSize="3xl" fontWeight="bold" color="gray.800" mt={2}>
            {value}
          </StatNumber>
          {helpText && (
            <StatHelpText fontSize="sm" color="gray.400" mt={1}>
              {helpText}
            </StatHelpText>
          )}
        </Box>
        <Box p={3} bg="blue.50" borderRadius="lg">
          <Icon size={24} color="#3182ce" />
        </Box>
      </Flex>
    </Stat>
  )

  if (link) {
    return (
      <Link to={link} style={{ textDecoration: 'none' }}>
        {content}
      </Link>
    )
  }

  return content
}

const Dashboard = () => {
  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 30000,
  })

  if (isLoading) {
    return (
      <Box>
        <Heading size="lg" mb={6}>Dashboard</Heading>
        <Text color="gray.500">Loading...</Text>
      </Box>
    )
  }

  if (error) {
    return (
      <Box>
        <Heading size="lg" mb={6}>Dashboard</Heading>
        <Text color="red.500">Error loading stats: {error.message}</Text>
      </Box>
    )
  }

  const totalDocs = stats?.documents?.total || 0
  const completedDocs = stats?.documents?.by_status?.completed || 0
  const totalChunks = stats?.chunks?.total || 0

  return (
    <Box>
      <Heading size="lg" mb={2}>Dashboard</Heading>
      <Text color="gray.500" mb={8}>Overview of your document Q&A system</Text>

      <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6} mb={8}>
        <StatCard
          title="Total Documents"
          value={totalDocs}
          icon={FiFileText}
          link="/documents"
          helpText={`${completedDocs} processed successfully`}
        />
        <StatCard
          title="Total Chunks"
          value={totalChunks}
          icon={FiLayers}
          helpText="Indexed for search"
        />
        <StatCard
          title="Go to Chat"
          value="→"
          icon={FiMessageCircle}
          link="/chat"
          helpText="Ask questions about your documents"
        />
      </SimpleGrid>

      <Box
        bg="blue.50"
        borderRadius="lg"
        p={6}
        borderLeft="4px solid"
        borderColor="blue.400"
      >
        <Heading size="md" mb={2} color="blue.800">
          Getting Started
        </Heading>
        <Text color="blue.700">
          1. Upload documents in the Documents section
          2. Wait for processing to complete
          3. Go to Chat to ask questions about your documents
        </Text>
      </Box>
    </Box>
  )
}

export default Dashboard
