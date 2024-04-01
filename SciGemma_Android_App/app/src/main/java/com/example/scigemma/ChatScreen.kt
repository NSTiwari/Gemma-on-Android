package com.example.scigemma

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel



@Composable
internal fun ChatRoute(
    chatViewModel: ChatViewModel = viewModel(
        factory = ChatViewModel.getFactory(LocalContext.current.applicationContext)
    )
) {
    val uiState by chatViewModel.uiState.collectAsStateWithLifecycle()
    val textInputEnabled by chatViewModel.isTextInputEnabled.collectAsStateWithLifecycle()
    ChatScreen(
        uiState,
        textInputEnabled
    ) { message ->
        chatViewModel.sendMessage(message)
    }
}

@Composable
fun ChatScreen(
    uiState: UiState,
    textInputEnabled: Boolean = true,
    onSendMessage: (String) -> Unit
) {
    var userMessage by rememberSaveable { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize(),
        verticalArrangement = Arrangement.Bottom
    ) {
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 8.dp),
            reverseLayout = true
        ) {
            items(uiState.messages) { chat ->
                ChatItem(chat)
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp, horizontal = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {

            Column { }

            Spacer(modifier = Modifier.width(8.dp))

            OutlinedTextField(
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color(0XFF6A1B9A),
                    unfocusedTextColor = Color(0XFF6A1B9A),
                    focusedBorderColor = Color(0XFF6A1B9A),
                    unfocusedBorderColor = Color(0XFF6A1B9A),
                    focusedLabelColor = Color(0XFF6A1B9A),
                    unfocusedLabelColor = Color(0XFF6A1B9A)
                ),
                value = userMessage,
                onValueChange = { userMessage = it },
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.Sentences,
                ),
                label = {
                    Text("Input")
                },
                modifier = Modifier
                    .weight(0.85f),
                enabled = textInputEnabled,
            )

            IconButton(
                onClick = {
                    if (userMessage.isNotBlank()) {
                        onSendMessage(userMessage)
                        userMessage = ""
                    }
                },
                modifier = Modifier
                    .padding(start = 16.dp)
                    .align(Alignment.CenterVertically)
                    .fillMaxWidth()
                    .weight(0.15f),
                enabled = textInputEnabled
            ) {
                Icon(
                    Icons.AutoMirrored.Default.Send,
                    contentDescription = "Send",
                    modifier = Modifier,
                    tint = Color(0XFF6A1B9A)
                )
            }
        }
    }
}

@Composable
fun ChatItem(
    chatMessage: ChatMessage
) {
    val backgroundColor = if (chatMessage.isFromUser) {
        Color(0XFF6A1B9A)
    } else {
        Color(0XFF9575CD)
    }

    val bubbleShape = if (chatMessage.isFromUser) {
        RoundedCornerShape(20.dp, 4.dp, 20.dp, 20.dp)
    } else {
        RoundedCornerShape(4.dp, 20.dp, 20.dp, 20.dp)
    }

    val horizontalAlignment = if (chatMessage.isFromUser) {
        Alignment.End
    } else {
        Alignment.Start
    }

    Column(
        horizontalAlignment = horizontalAlignment,
        modifier = Modifier
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .fillMaxWidth()
    ) {
        val author = if (chatMessage.isFromUser) {
            "User"
        } else {
            "Model"
        }
        Text(
            color = Color(0xFF212121),
            text = author,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 4.dp)
        )
        Row {
            BoxWithConstraints {
                Card(
                    colors = CardDefaults.cardColors(containerColor = backgroundColor),
                    shape = bubbleShape,
                    modifier = Modifier.widthIn(0.dp, maxWidth * 0.9f)
                ) {
                    if (chatMessage.isLoading) {
                        CircularProgressIndicator(
                            color = Color(0XFF6A1B9A),
                            modifier = Modifier.padding(16.dp)
                        )
                    } else {
                        var response = chatMessage.message
                        response = response.replace("<start_of_turn>", "")
                        response = response.replace("</start_of_turn>", "")
                        response = response.replace("<end_of_turn", "")
                        response = response.replace("</end_of_turn>", "")
                        response = response.replace("impra ", "")
                        response = response.replace("impractically ", "")
                        response = response.replace("reluct ", "")
                        response = response.replace("modelAnswer:", "")
                        response = response.replace("userAnswer:", "")
                        response = response.replace("encomp ", "")
                        response = response.replace("encompiring ", "")
                        response = response.replace("encomprifying ", "")
                        response = response.replace("increa ", "")
                        response = response.replace("maneuv ", "")
                        response = response.replace("guarante ", "")

                        Text(
                                text = response,
                                modifier = Modifier.padding(16.dp)
                        )
                    }
                }
            }
        }
    }
}
