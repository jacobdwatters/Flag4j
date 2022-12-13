package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.SparseTensorBase;

import java.util.Arrays;


/**
 * Complex sparse tensor.
 */
public class SparseCTensor extends SparseTensorBase<CNumber[]> {


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public SparseCTensor(Shape shape) {
        super(shape, 0, new CNumber[0], new int[0][0]);
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public SparseCTensor(Shape shape, double[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);

        for(int i=0; i<indices.length; i++) {
            super.entries[i] = new CNumber(nonZeroEntries[i]);
        }
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public SparseCTensor(Shape shape, int[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);

        for(int i=0; i<indices.length; i++) {
            super.entries[i] = new CNumber(nonZeroEntries[i]);
        }
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public SparseCTensor(Shape shape, CNumber[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
    }


//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, double[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.clone());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//            super.indices[i] = indices.get(i);
//        }
//    }
//
//
//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, int[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.clone());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//            super.indices[i] = indices.get(i);
//        }
//    }
//
//
//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, CNumber[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i].re!=0 || entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.clone());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = nonZeroEntries.get(i).clone();
//            super.indices[i] = indices.get(i);
//        }
//    }


    /**
     * Constructs a sparse complex tensor whose non-zero values, indices, and shape are specified by another sparse complex
     * tensor.
     * @param A
     */
    public SparseCTensor(SparseCTensor A) {
        super(A.shape.clone(), A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
        for(int i=0; i<indices.length; i++) {
            super.indices[i] = A.indices[i].clone();
        }
    }
}