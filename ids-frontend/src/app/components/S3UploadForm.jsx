"use client"
import React from 'react'
import { useState } from 'react'

const S3UploadForm = () => {
    const [file, setFile] = useState(null)
    const [Uploading, setUploading] = useState(false)

    const handleFileChange = (e) => {
        setFile(e.target.files[0])
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        if(!file) return;

        setUploading(true)
        const formData = new FormData()
        formData.append("file", file)
        try {
            const response = await fetch("/api/s3-upload", {
                method: "POST",
                body: formData
            })
            const data = await response.json()
            console.log(data.status)
            setUploading(false)
        } catch (error) {
            console.error(error)
            setUploading(false)
    }
    }

  return (
    <div>
        <form onSubmit={handleSubmit}>
            <input type="file" accept='text/txt' onChange={handleFileChange}/>
            <button type="submit" disabled={!file || Uploading}>
                {Uploading ? "Uploading..." : "Upload"}
            </button>
        </form>
    </div>
  )
}

export default S3UploadForm