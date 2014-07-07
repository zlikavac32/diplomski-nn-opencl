package com.msuflaj.trainer;

public class UnexpectedNetworkException extends Exception {

    public UnexpectedNetworkException(String message) {
        super(message);
    }

    public UnexpectedNetworkException(String message, Throwable cause) {
        super(message, cause);
    }

    public UnexpectedNetworkException(Throwable cause) {
        super(cause);
    }

    public UnexpectedNetworkException() { }
}
