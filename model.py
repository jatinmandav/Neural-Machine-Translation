import keras
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers import Activation, dot, concatenate
from keras.models import Model

class NMTModelDef:
    def __init__(self, latent_dim, num_encoder_tokens, num_decoder_tokens,
                    encoder_embedding, decoder_embedding, pretrained_path=None):

        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding

        self.pretrained_path = pretrained_path

    def build(self, inference=False):
        # ----------------------- #
        # NMT model for Training  #
        # ----------------------- #

        # Encoder
        encoder_input = Input(shape=(None, ))
        encoder_embedding = Embedding(self.num_encoder_tokens, self.latent_dim, mask_zero=True,
                                    weights=[self.encoder_embedding], name='encoder_embedding')(encoder_input)
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='encoder')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_input = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.latent_dim, mask_zero=True,
                                    weights=[self.decoder_embedding], name='decoder_embedding')(decoder_input)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Luong's Attention
        attention = dot([encoder_outputs, decoder_outputs], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        context = dot([attention, encoder_outputs], name='context_vector', axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_outputs], name='decoder_combined_context_vector')

        # Output
        out_dense = Dense(self.num_decoder_tokens, activation='softmax', name='output')
        output_training = out_dense(decoder_combined_context)

        self.training_model = Model([encoder_input, decoder_input], output_training)
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        # ----------------------- #
        # NMT model for Inference #
        # ----------------------- #

        if inference:
            if self.pretrained_path:
                self.training_model.load_weights(self.pretrained_path)

            # Encoder
            self.encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

            # Decoder
            decoder_state_input_h = Input(shape=(None, ))
            decoder_state_input_c = Input(shape=(None, ))
            encoder_inference_input = Input(shape=(None, self.latent_dim))

            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_embedding_inference = decoder_embedding(decoder_input)

            decoder_output_inference, inf_state_h, inf_state_c = decoder_lstm(decoder_embedding_inference, initial_state=decoder_states_inputs)
            decoder_states = [inf_state_h, inf_state_c]

            # Luong's Attention
            attention = dot([decoder_output_inference, encoder_inference_input], axes=(2, 2))
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder_inference_input], axes=[2, 1], name='context_vector')
            decoder_combined_context = concatenate([context, decoder_output_inference], name='decoder_combined_context_vector')

            output_inference = out_dense(decoder_combined_context)

            self.decoder_model = Model([decoder_input, encoder_inference_input] + decoder_states_inputs,
                                       [output_inference] + decoder_states)

            return self.training_model, self.encoder_model, self.decoder_model

        else:
            return self.training_model
