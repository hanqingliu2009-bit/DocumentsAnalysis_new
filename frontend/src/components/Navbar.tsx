import { Box, Flex, Link, Text, VStack, Icon } from '@chakra-ui/react'
import { FiFileText, FiHome, FiMessageSquare } from 'react-icons/fi'
import { Link as RouterLink, useLocation } from 'react-router-dom'

const NavItem = ({ icon: Icon, to, children }: { icon: any; to: string; children: React.ReactNode }) => {
  const location = useLocation()
  const isActive = location.pathname === to

  return (
    <Link
      as={RouterLink}
      to={to}
      p={3}
      borderRadius="md"
      bg={isActive ? 'blue.500' : 'transparent'}
      color={isActive ? 'white' : 'gray.300'}
      _hover={{ bg: isActive ? 'blue.600' : 'gray.700' }}
      textDecoration="none"
      display="block"
    >
      <Flex align="center" gap={3}>
        <Icon size={20} />
        <Text fontWeight="medium">{children}</Text>
      </Flex>
    </Link>
  )
}

const Navbar = () => {
  return (
    <Flex
      w="240px"
      flexShrink={0}
      alignSelf="stretch"
      bg="gray.800"
      color="white"
      direction="column"
      p={4}
    >
      <Text fontSize="xl" fontWeight="bold" mb={8} color="blue.400">
        Document Q&A
      </Text>

      <VStack align="stretch" gap={2} flex={1}>
        <NavItem icon={FiHome} to="/">Dashboard</NavItem>
        <NavItem icon={FiFileText} to="/documents">Documents</NavItem>
        <NavItem icon={FiMessageSquare} to="/chat">Chat</NavItem>
      </VStack>

      <Text fontSize="xs" color="gray.500" mt={4}>
        v0.1.0
      </Text>
    </Flex>
  )
}

export default Navbar
