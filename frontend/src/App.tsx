import { Box, Flex } from '@chakra-ui/react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import DocumentManager from './pages/DocumentManager'
import ChatInterface from './pages/ChatInterface'

function App() {
  return (
    <BrowserRouter>
      <Flex h="100vh">
        <Navbar />
        <Box flex={1} p={6} overflow="auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/documents" element={<DocumentManager />} />
            <Route path="/chat" element={<ChatInterface />} />
          </Routes>
        </Box>
      </Flex>
    </BrowserRouter>
  )
}

export default App
