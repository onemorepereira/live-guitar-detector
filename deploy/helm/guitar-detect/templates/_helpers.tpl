{{/*
Expand the name of the chart.
*/}}
{{- define "guitar-detect.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a fully qualified app name.
*/}}
{{- define "guitar-detect.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "guitar-detect.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "guitar-detect.labels" -}}
helm.sh/chart: {{ include "guitar-detect.chart" . }}
{{ include "guitar-detect.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "guitar-detect.selectorLabels" -}}
app.kubernetes.io/name: {{ include "guitar-detect.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Per-component selector labels. Usage:
  {{ include "guitar-detect.componentLabels" (dict "ctx" . "component" "gateway") }}
*/}}
{{- define "guitar-detect.componentLabels" -}}
{{ include "guitar-detect.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "guitar-detect.componentSelectorLabels" -}}
{{ include "guitar-detect.selectorLabels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}
