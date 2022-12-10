package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.TensorBase;
import com.flag4j.util.ErrorMessages;

/**
 * Complex dense tensor.
 */
public class CTensor extends TensorBase<CNumber[]> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = new CNumber();
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = new CNumber(fillValue);
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, CNumber fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = fillValue.clone();
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = new CNumber(entries[i]);
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, int[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = new CNumber(entries[i]);
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * Note, unlike other constructors, the entries parameter is not copied.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, CNumber[] entries) {
        super(shape, entries);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(Tensor A) {
        super(A.shape.clone(), new CNumber[A.totalEntries().intValue()]);
        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = new CNumber(A.entries[i]);
        }
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(CTensor A) {
        super(A.shape.clone(), new CNumber[A.totalEntries().intValue()]);
        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = A.entries[i].clone();
        }
    }
}
