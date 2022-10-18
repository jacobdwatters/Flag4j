package com.flag4j.complex_numbers;


/**
 * A CNumberToken is the smallest unit of a string which is being parsed to a complex number.
 */
class CNumberToken {
    private String kind;
    private String details;


    /**
     * Creates a token of the specified kind and with the specified details.
     * @param k Kind of token to create.
     * @param d Details of the token.
     */
    protected CNumberToken( String k, String d ) {
        kind = k;  details = d;
    }


    /**
     * Checks if token is of kind s.
     *
     * @param s token of interest.
     * @return true if token kind equalTo s.
     */
    protected boolean isKind( String s ) {
        return kind.equals( s );
    }


    /**
     * Gets the kind of this token.
     * @return token kind.
     */
    protected String getKind() { return kind; }


    /**
     * Gets the details of this token.
     * @return token details.
     */
    protected String getDetails() { return details; }


    /**
     * Checks if given tokens kind and details match k and d respectively.
     *
     * @param k token kind.
     * @param d token details.
     * @return True if token matches kind and details, otherwise false.
     */
    protected boolean matches( String k, String d ) {
        return kind.equals(k) && details.equals(d);
    }


    /**
     * If a given tokens kind and details match k and d respectively then the program will halt.
     *
     * @param k token kind
     * @param d token details
     */
    protected void errorCheck(String k, String d) {
        if(!this.matches(k, d)) {
            System.out.println("Expecting token [" + k + "," + d + "] but got " + this);
            System.exit(1);
        }
    }


    /**
     * If a given token does not match the provided kind then the program will halt.
     *
     * @param k token kind.
     */
    protected void errorCheck(String k) {
        if(!this.kind.equals(k)) {
            System.out.println("Expecting token of kind " + k + " but got " + this);
            System.exit(1);
        }
    }


    /**
     * Constructs a string representation of a token. This will be of the form [kind, details].
     * @return A string representing of this token.
     */
    public String toString() {
        return "[" + this.kind + "," + this.details + "]";
    }
}
