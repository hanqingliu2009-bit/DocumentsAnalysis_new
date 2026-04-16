import { Box, Button, Flex, Heading, IconButton, Input, Spinner, Text, VStack, HStack, Avatar, Divider } from '@chakra-ui/react'
import { useState, useRef, useEffect } from 'react'
import { FiSend, FiTrash2, FiUser } from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'

import axios from 'axios'

import { queryChat } from '../services/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: {
    chunk_id: string
    text: string
    score: number
  }[]
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const messageText = input.trim()

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageText,
    }

    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }))

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const data = await queryChat(messageText, historyPayload)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.message,
        sources: data.sources,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error: unknown) {
      let detail =
        'Sorry, I encountered an error while processing your request. Please try again.'
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          detail =
            'Request timed out. The server may still be loading the embedding model or calling the LLM; wait and try again, or check backend logs.'
        } else {
          const data = error.response?.data as { detail?: unknown } | undefined
          const serverDetail = data?.detail
          if (typeof serverDetail === 'string' && serverDetail.trim()) {
            detail = serverDetail
          }
        }
      }
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: detail,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      setMessages([])
    }
  }

  return (
    <Flex h="calc(100vh - 48px)" direction="column">
      <Flex justify="space-between" align="center" mb={4}>
        <Box>
          <Heading size="lg">Chat</Heading>
          <Text color="gray.500" fontSize="sm">
            Ask questions about your documents
          </Text>
        </Box>
        {messages.length > 0 && (
          <IconButton
            aria-label="Clear conversation"
            icon={<FiTrash2 />}
            onClick={handleClear}
            variant="ghost"
            colorScheme="red"
          />
        )}
      </Flex>

      <Box
        flex={1}
        overflow="auto"
        bg="gray.50"
        borderRadius="lg"
        p={4}
        mb={4}
      >
        {messages.length === 0 ? (
          <Flex
            h="100%"
            direction="column"
            justify="center"
            align="center"
            color="gray.400"
          >
            <Text fontSize="lg" mb={2}>
              Start a conversation
            </Text>
            <Text fontSize="sm">
              Ask questions about your uploaded documents
            </Text>
          </Flex>
        ) : (
          <VStack spacing={4} align="stretch">
            {messages.map((message) => (
              <Box
                key={message.id}
                alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                maxW="80%"
              >
                <HStack
                  spacing={2}
                  align="flex-start"
                  justify={message.role === 'user' ? 'flex-end' : 'flex-start'}
                >
                  {message.role === 'assistant' && (
                    <Avatar size="sm" name="AI" bg="blue.500" />
                  )}
                  <Box
                    bg={message.role === 'user' ? 'blue.500' : 'white'}
                    color={message.role === 'user' ? 'white' : 'gray.800'}
                    borderRadius="lg"
                    p={3}
                    boxShadow="sm"
                    borderWidth={message.role === 'user' ? 0 : 1}
                    borderColor="gray.200"
                  >
                    <Box
                      css={{
                        '& p': { marginBottom: '0.5rem' },
                        '& p:last-child': { marginBottom: 0 },
                        '& ul, & ol': { marginLeft: '1rem', marginBottom: '0.5rem' },
                      }}
                    >
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </Box>
                  </Box>
                  {message.role === 'user' && (
                    <Avatar size="sm" icon={<FiUser />} bg="gray.500" />
                  )}
                </HStack>
                {message.sources && message.sources.length > 0 && (
                  <Box mt={2} ml={10}>
                    <Text fontSize="xs" color="gray.500" mb={1}>
                      Sources:
                    </Text>
                    {message.sources.map((source, idx) => (
                      <Box
                        key={idx}
                        bg="gray.100"
                        p={2}
                        borderRadius="md"
                        mt={1}
                        fontSize="xs"
                      >
                        <Text color="gray.600" noOfLines={2}>
                          {source.text}
                        </Text>
                        <Text color="gray.400" mt={1}>
                          Score: {source.score.toFixed(3)}
                        </Text>
                      </Box>
                    ))}
                  </Box>
                )}
              </Box>
            ))}
            <div ref={messagesEndRef} />
          </VStack>
        )}
      </Box>

      {isLoading && (
        <HStack spacing={3} py={2} px={1} color="gray.600">
          <Spinner size="sm" color="blue.500" />
          <Text fontSize="sm">
            Waiting for the server. The first reply can take several minutes while the embedding model loads
            (one-time). Watch the backend terminal for progress or errors.
          </Text>
        </HStack>
      )}

      <HStack spacing={2}>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          size="lg"
          disabled={isLoading}
        />
        <IconButton
          aria-label="Send message"
          icon={isLoading ? <Spinner /> : <FiSend />}
          onClick={handleSend}
          isDisabled={!input.trim() || isLoading}
          size="lg"
          colorScheme="blue"
        />
      </HStack>
    </Flex>
  )
}

export default ChatInterface
